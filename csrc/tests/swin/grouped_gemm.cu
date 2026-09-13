#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "swin/window_attention/kernel/default_grouped_gemm.h"
#include "test_utils.h"

using tiny_cutlass::testing::DeviceBuffer;

void check(bool condition, char const* message) {
  if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void check(cutlass::Status status) {
  if (status != cutlass::Status::kSuccess) throw std::runtime_error(cutlass::cutlassGetStatusString(status));
}

template <typename ElementA, typename ElementB>
void run(int iterations, char const* label) {
  using Policy = tiny_cutlass::swin::window_attention::kernel::DefaultGroupedGemm<
      cutlass::arch::Sm89, ElementA, ElementB>;
  using Gemm = typename Policy::DeviceOperator;
  // Small window products, channel projections, and ragged grouped shapes.
  // N is divisible by the 4-element epilogue access; K by the 16-byte input access.
  std::vector<cutlass::gemm::GemmCoord> shapes = {
      {16, 16, 32}, {16, 32, 16}, {16, 64, 96}, {31, 68, 80}, {128, 96, 192}, {65, 128, 512}};
  int groups = int(shapes.size());
  std::vector<std::unique_ptr<DeviceBuffer<ElementA>>> a;
  std::vector<std::unique_ptr<DeviceBuffer<ElementB>>> b;
  std::vector<std::unique_ptr<DeviceBuffer<float>>> c, d;
  std::vector<ElementA*> pa;
  std::vector<ElementB*> pb;
  std::vector<float*> pc, pd;
  std::vector<int64_t> lda, ldb, ldc;
  std::vector<std::vector<float>> refs;
  float scale_a = 0.25f, scale_b = 0.125f;
  float alpha = scale_a * scale_b, beta = 0.5f;
  for (int g = 0; g < groups; ++g) {
    auto s = shapes[g];
    std::vector<float> original_a(size_t(s.m()) * s.k()), original_b(size_t(s.n()) * s.k()), source(size_t(s.m()) * s.n());
    tiny_cutlass::testing::fill_random_uniform(original_a, 2026 + g, -1.f, 1.f);
    tiny_cutlass::testing::fill_random_uniform(original_b, 2046 + g, -1.f, 1.f);
    tiny_cutlass::testing::fill_random_uniform(source, 2066 + g, -1.f, 1.f);
    std::vector<ElementA> qa(original_a.size());
    std::vector<ElementB> qb(original_b.size());
    for (size_t i = 0; i < qa.size(); ++i) qa[i] = ElementA(original_a[i] / scale_a);
    for (size_t i = 0; i < qb.size(); ++i) qb[i] = ElementB(original_b[i] / scale_b);
    refs.emplace_back(source.size());
    // FP64 oracle over the actual FP8 operands: quantization error must not be
    // mistaken for a GEMM implementation error.
    for (int m = 0; m < s.m(); ++m) for (int n = 0; n < s.n(); ++n) {
      double acc = 0;
      for (int k = 0; k < s.k(); ++k) acc += double(float(qa[size_t(m)*s.k()+k])) * float(qb[size_t(n)*s.k()+k]);
      refs.back()[size_t(m)*s.n()+n] = float(alpha * acc + beta * source[size_t(m)*s.n()+n]);
    }
    a.emplace_back(std::make_unique<DeviceBuffer<ElementA>>(qa.size()));
    b.emplace_back(std::make_unique<DeviceBuffer<ElementB>>(qb.size()));
    c.emplace_back(std::make_unique<DeviceBuffer<float>>(source.size()));
    d.emplace_back(std::make_unique<DeviceBuffer<float>>(source.size()));
    check(a.back()->get() && b.back()->get() && c.back()->get() && d.back()->get(), "allocation failed");
    check(a.back()->copy_from_host(qa) && b.back()->copy_from_host(qb) && c.back()->copy_from_host(source), "upload failed");
    check(cudaMemset(d.back()->get(), 0xff, source.size()*sizeof(float)));
    pa.push_back(a.back()->get()); pb.push_back(b.back()->get());
    pc.push_back(c.back()->get()); pd.push_back(d.back()->get());
    lda.push_back(s.k()); ldb.push_back(s.k()); ldc.push_back(s.n());
  }
  DeviceBuffer<cutlass::gemm::GemmCoord> ds(groups);
  DeviceBuffer<ElementA*> da(groups);
  DeviceBuffer<ElementB*> db(groups);
  DeviceBuffer<float*> dc(groups), dd(groups);
  DeviceBuffer<int64_t> dla(groups), dlb(groups), dlc(groups);
  check(ds.copy_from_host(shapes) && da.copy_from_host(pa) && db.copy_from_host(pb) && dc.copy_from_host(pc) && dd.copy_from_host(pd) &&
        dla.copy_from_host(lda) && dlb.copy_from_host(ldb) && dlc.copy_from_host(ldc), "descriptor upload failed");
  int blocks = Gemm::sufficient(shapes.data(), groups);
  check(blocks > 0, "unsupported kernel resources");
  typename Gemm::Arguments args(ds.get(), groups, blocks, {alpha, beta}, da.get(), db.get(), dc.get(), dd.get(),
                                dla.get(), dlb.get(), dlc.get(), dlc.get(), shapes.data());
  DeviceBuffer<uint8_t> workspace(Gemm::get_workspace_size(args));
  Gemm gemm;
  cudaStream_t stream;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  check(gemm.initialize(args, workspace.get(), stream));
  check(gemm.run(stream));
  check(cudaStreamSynchronize(stream));
  for (int g = 0; g < groups; ++g) {
    std::vector<float> actual(refs[g].size());
    check(d[g]->copy_to_host(actual), "download failed");
    auto result = tiny_cutlass::testing::compare_host(actual, refs[g]);
    std::cout << label << " group=" << g << " MAE=" << result.mae << " max_abs=" << result.max_abs << '\n';
    // Native FP8 MMA rounding/subnormal behavior need not match the FP64 oracle.
    // Keep the repository MAE gate and an explicit max-error bound.
    check(result.finite && result.mae <= 1e-3 && result.max_abs <= 1e-2, "grouped reference parity failed");
  }
  cudaGraph_t graph;
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  check(gemm.run(stream));
  check(cudaStreamEndCapture(stream, &graph));
  size_t nodes = 0;
  check(cudaGraphGetNodes(graph, nullptr, &nodes));
  check(nodes == 1, "grouped GEMM must be one kernel launch");
  check(cudaGraphDestroy(graph));
  if (iterations > 0) {
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
    for (int i=0; i<10; ++i) check(gemm.run(stream));
    check(cudaEventRecord(start, stream));
    for (int i=0; i<iterations; ++i) check(gemm.run(stream));
    check(cudaEventRecord(stop, stream)); check(cudaEventSynchronize(stop));
    float ms;
    check(cudaEventElapsedTime(&ms, start, stop));
    std::cout << label << " Runtime: " << ms/iterations << " ms (six grouped products)\n";
    check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
  }
  check(cudaStreamDestroy(stream));
}

int main(int argc, char** argv) try {
  int iterations = 0;
  if (argc == 3 && std::string(argv[1]) == "--iterations") iterations = std::stoi(argv[2]);
  else check(argc == 1, "usage: swin_grouped_gemm [--iterations N]");
  check(iterations >= 0, "iterations must be nonnegative");
  int device;
  cudaDeviceProp properties;
  check(cudaGetDevice(&device)); check(cudaGetDeviceProperties(&properties, device));
  check(properties.major == 8 && properties.minor == 9, "this validation requires SM89; no fallback");
  std::cout << "Device: " << properties.name << " SM89\n";
  run<cutlass::float_e4m3_t, cutlass::float_e4m3_t>(iterations, "e4m3/e4m3");
  run<cutlass::float_e5m2_t, cutlass::float_e5m2_t>(iterations, "e5m2/e5m2");
  std::cout << "PASS: 12 FP8 grouped GEMMs, one launch per group list\n";
  return 0;
} catch (std::exception const& e) {
  std::cerr << "FAIL: " << e.what() << '\n';
  return 1;
}
