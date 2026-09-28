#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "cutlass/half.h"
#include "swin/fused_swin_layer/device/fused_swin_layer.h"
#include "swin/window_attention/window_index.h"
#include "swin/window_attention/position_bias.h"
#include "test_utils.h"

namespace wa = tiny_cutlass::swin::window_attention;
using Element = cutlass::half_t;
namespace fused = tiny_cutlass::swin::fused_swin_layer;
#if FUSED_SWIN_ARCH == 80
using Policy = fused::kernel::DefaultFusedSwinLayer<cutlass::arch::Sm80, Element>;
#elif FUSED_SWIN_ARCH == 89
using Policy = fused::kernel::DefaultFusedSwinLayer<cutlass::arch::Sm89, Element>;
#else
#error Unsupported architecture
#endif
using Device = fused::device::FusedSwinLayer<Policy>;
using tiny_cutlass::testing::DeviceBuffer;

void require(bool ok, char const* what) { if (!ok) throw std::runtime_error(what); }
void cuda_check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void cutlass_check(cutlass::Status status) {
  if (status != cutlass::Status::kSuccess) throw std::runtime_error(cutlass::cutlassGetStatusString(status));
}

template <typename T>
std::vector<T> read(std::filesystem::path const& path, size_t count) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  require(bool(file) && size_t(file.tellg()) == count * sizeof(T), "missing or incorrectly sized fixture");
  file.seekg(0);
  std::vector<T> data(count);
  require(bool(file.read(reinterpret_cast<char*>(data.data()), count * sizeof(T))), "fixture read failed");
  return data;
}

// The operator must reject unsupported options instead of selecting a fallback.
void contracts(Device::Arguments const& good) {
  auto reject = [](auto const& a) {
    require(Device::can_implement(a) != cutlass::Status::kSuccess, "invalid configuration accepted");
  };
  auto a = good; a.problem.window_size = 7; reject(a);
  a = good; a.problem.quant_fp8 = true; reject(a);
  a = good; a.problem.attn_drop = .1f; reject(a);
  a = good; a.problem.proj_drop = .1f; reject(a);
  a = good; a.problem.qk_channels = 7; reject(a);
  a = good; a.problem.value_channels = 72; reject(a);
  a = good; a.problem.groups = 65; reject(a);
  a = good; a.problem.channels = 0; reject(a);
  a = good; a.problem.height = -1; reject(a);
  a = good; a.problem.shift_h = 4; reject(a);
  a = good; a.problem.shift_w = -1; reject(a);
  a = good; a.problem.rms_epsilon = std::numeric_limits<float>::quiet_NaN(); reject(a);
  a = good; a.problem.rms_epsilon = 0; reject(a);
  a = good; a.problem.rms_epsilon = std::numeric_limits<float>::infinity(); reject(a);
  a = good; a.rms_weight = nullptr; reject(a);
  a = good; a.problem.batch = 65535; a.problem.height = 1048576; reject(a);
  a = good; a.problem.mlp_ratio = 3; reject(a);
  a = good; a.mlp_rms_weight = nullptr; reject(a);
  a = good; a.fc1_weight = nullptr; reject(a);
  a = good; a.fc2_bias = nullptr; reject(a);
  a = good; a.problem.mlp_rms_epsilon = 0; reject(a);
  a = good; a.problem.channels = 64; reject(a);
  a = good; a.problem.groups = 2; reject(a);
  a = good; a.input = nullptr; reject(a);
  a = good; a.gather = nullptr; reject(a);
  a = good; a.scatter = nullptr; reject(a);
  a = good; a.output = const_cast<Element*>(a.input); reject(a);
  a = good; a.input = reinterpret_cast<Element const*>(reinterpret_cast<char const*>(a.input) + 1); reject(a);
}

struct Stream {
  cudaStream_t value = nullptr;
  Stream() { cuda_check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(value); }
};
struct Event {
  cudaEvent_t value = nullptr;
  Event() { cuda_check(cudaEventCreate(&value)); }
  ~Event() { cudaEventDestroy(value); }
};

int main(int argc, char** argv) try {
  std::filesystem::path dir;
  int iterations = 0;
  bool graph = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--case-dir" && i + 1 < argc) dir = argv[++i];
    else if (arg == "--iterations" && i + 1 < argc) iterations = std::stoi(argv[++i]);
    else if (arg == "--graph") graph = true;
    else throw std::runtime_error("usage: fused_swin_layer --case-dir DIR [--iterations N] [--graph]");
  }
  require(!dir.empty() && iterations >= 0, "case directory and nonnegative iterations required");
  wa::WindowAttentionProblem p;
  int separate, bias_enabled, relative;
  std::ifstream problem(dir / "problem.txt");
  require(bool(problem >> p.batch >> p.height >> p.width >> p.channels >> p.groups >> p.qk_channels >>
      p.value_channels >> p.shift_h >> p.shift_w >> separate >> p.rms_epsilon >> bias_enabled >> relative),
      "invalid problem.txt");
  p.qk_mode = separate ? wa::QkMode::kSeparate : wa::QkMode::kShared;
  require(p.valid(), "unsupported fixture problem");
  auto indices = wa::build_window_attention_index(p);
  // Every output token has exactly one writer, including reflected small maps.
  std::vector<int> coverage(size_t(p.tokens()), 0);
  for (size_t row = 0; row < indices.gather.size(); ++row) {
    require(indices.gather[row] >= 0 && indices.gather[row] < p.tokens(), "gather outside input");
    if (indices.scatter[row] >= 0) {
      require(indices.scatter[row] < p.tokens(), "scatter outside output");
      ++coverage[indices.scatter[row]];
    }
  }
  for (int count : coverage) require(count == 1, "scatter must cover each output exactly once");
  size_t count = size_t(p.tokens()) * p.channels;
  auto x = read<Element>(dir / "input.bin", count);
  auto rms = read<Element>(dir / "rms_weight.bin", p.channels);
  auto qkv = read<Element>(dir / "qkv_weight.bin", size_t(p.groups) * p.projected_channels() * p.channels);
  auto pos = read<Element>(dir / "position_bias.bin", size_t(p.groups) * 256);
  if (relative) {
    auto table = read<Element>(dir / "relative_bias.bin", size_t(p.groups) * 49);
    auto expanded = wa::expand_relative_position_bias(table.data(), p.groups);
    require(expanded == pos, "precomputed relative bias differs from Python fixture");
    pos = std::move(expanded);
  }
  auto weight = read<Element>(dir / "output_weight.bin", size_t(p.channels) * p.hidden_channels());
  auto bias = read<Element>(dir / "output_bias.bin", p.channels);
  auto reference = read<float>(dir / "reference.bin", count);
  DeviceBuffer<Element> dx(x.size()), drms(rms.size()), dqkv(qkv.size()), dpos(pos.size()), dw(weight.size()), db(bias.size()), dy(count);
  DeviceBuffer<int> dg(indices.gather.size()), ds(indices.scatter.size());
  require(dx.get() && drms.get() && dqkv.get() && dpos.get() && dw.get() && db.get() && dy.get() && dg.get() && ds.get(), "allocation failed");
  require(dx.copy_from_host(x) && drms.copy_from_host(rms) && dqkv.copy_from_host(qkv) && dpos.copy_from_host(pos) &&
      dw.copy_from_host(weight) && db.copy_from_host(bias) && dg.copy_from_host(indices.gather) &&
      ds.copy_from_host(indices.scatter), "upload failed");
  Device::Arguments a;
  a.problem.batch = p.batch;
  a.problem.height = p.height;
  a.problem.width = p.width;
  a.problem.channels = p.channels;
  a.problem.groups = p.groups;
  a.problem.qk_channels = p.qk_channels;
  a.problem.value_channels = p.value_channels;
  a.problem.window_size = p.window_size;
  a.problem.shift_h = p.shift_h;
  a.problem.shift_w = p.shift_w;
  a.problem.rms_epsilon = p.rms_epsilon;
  a.problem.separate_qk = separate; a.input = dx.get(); a.rms_weight = drms.get(); a.qkv_weight = dqkv.get(); a.position_bias = dpos.get();
  a.output_weight = dw.get(); a.output_bias = bias_enabled ? db.get() : nullptr;
  a.gather = dg.get(); a.scatter = ds.get(); a.output = dy.get();
  std::ifstream mlp_problem(dir / "mlp.txt");
  require(bool(mlp_problem >> a.problem.mlp_ratio >> a.problem.mlp_rms_epsilon), "invalid mlp.txt");
  auto mlp_rms = read<Element>(dir / "mlp_rms_weight.bin", p.channels);
  auto fc1 = read<Element>(dir / "fc1_weight.bin", p.channels * p.channels * a.problem.mlp_ratio);
  auto b1 = read<Element>(dir / "fc1_bias.bin", p.channels * a.problem.mlp_ratio);
  auto fc2 = read<Element>(dir / "fc2_weight.bin", p.channels * p.channels * a.problem.mlp_ratio);
  auto b2 = read<Element>(dir / "fc2_bias.bin", p.channels);
  DeviceBuffer<Element> dmr(mlp_rms.size()), df1(fc1.size()), db1(b1.size()), df2(fc2.size()), db2(b2.size());
  require(dmr.get() && df1.get() && db1.get() && df2.get() && db2.get(), "MLP allocation failed");
  require(dmr.copy_from_host(mlp_rms) && df1.copy_from_host(fc1) && db1.copy_from_host(b1) &&
      df2.copy_from_host(fc2) && db2.copy_from_host(b2), "MLP upload failed");
  a.mlp_rms_weight = dmr.get(); a.fc1_weight = df1.get(); a.fc1_bias = db1.get();
  a.fc2_weight = df2.get(); a.fc2_bias = db2.get();
  contracts(a);
  Device operation;
  require(operation.run() != cutlass::Status::kSuccess, "uninitialized run accepted");
  require(Device::get_workspace_size(a) == 0, "unexpected workspace");
  int device = 0; cudaDeviceProp props{};
  cuda_check(cudaGetDevice(&device)); cuda_check(cudaGetDeviceProperties(&props, device));
  std::cout << "Device: " << props.name << " CC=" << props.major << props.minor
            << " shared_bytes=" << sizeof(typename Policy::CutlassKernel::SharedStorage) << '\n';
  Stream stream;
  cuda_check(cudaMemsetAsync(dy.get(), 0xff, count * sizeof(Element), stream.value));
  cutlass_check(operation(a, stream.value));
  cuda_check(cudaStreamSynchronize(stream.value));
  auto verify = [&]() {
    std::vector<Element> actual(count);
    require(dy.copy_to_host(actual), "download failed");
    auto error = tiny_cutlass::testing::compare_host(actual, reference);
    std::cout << "MAE=" << error.mae << " max_abs=" << error.max_abs << " finite=" << error.finite << '\n';
    require(error.finite && error.mae <= 1e-3 && error.max_abs <= 2e-2, "reference parity failed");
    std::ofstream out(dir / "actual.bin", std::ios::binary);
    require(bool(out.write(reinterpret_cast<char const*>(actual.data()), actual.size() * sizeof(Element))), "output write failed");
  };
  verify();
  if (graph) {
    cudaGraph_t captured;
    cudaGraphExec_t executable;
    cuda_check(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
    cutlass_check(operation(a, stream.value));
    cuda_check(cudaStreamEndCapture(stream.value, &captured));
    size_t nodes = 0;
    cuda_check(cudaGraphGetNodes(captured, nullptr, &nodes));
    require(nodes == 1, "block must capture exactly one kernel node");
    cuda_check(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
    cuda_check(cudaMemsetAsync(dy.get(), 0xff, count * sizeof(Element), stream.value));
    cuda_check(cudaGraphLaunch(executable, stream.value));
    cuda_check(cudaStreamSynchronize(stream.value));
    verify();
    cuda_check(cudaGraphExecDestroy(executable));
    cuda_check(cudaGraphDestroy(captured));
    std::cout << "CUDA graph: one kernel node, replay parity passed\n";
  }
  // Timing is unreachable when any preceding correctness check fails.
  if (iterations) {
    cudaGraph_t captured;
    cudaGraphExec_t executable;
    cuda_check(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
    cutlass_check(operation(a, stream.value));
    cuda_check(cudaStreamEndCapture(stream.value, &captured));
    cuda_check(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
    for (int i = 0; i < 20; ++i) cuda_check(cudaGraphLaunch(executable, stream.value));
    Event start, stop;
    cuda_check(cudaEventRecord(start.value, stream.value));
    for (int i = 0; i < iterations; ++i) cuda_check(cudaGraphLaunch(executable, stream.value));
    cuda_check(cudaEventRecord(stop.value, stream.value));
    cuda_check(cudaEventSynchronize(stop.value));
    float elapsed;
    cuda_check(cudaEventElapsedTime(&elapsed, start.value, stop.value));
    std::cout << "Runtime: " << elapsed / iterations << " ms\n";
    std::cout << "Timing: full_block_cuda_graph\n";
    cuda_check(cudaGraphExecDestroy(executable)); cuda_check(cudaGraphDestroy(captured));
  }
  std::cout << "PASS " << dir.filename().string() << '\n';
  return 0;
} catch (std::exception const& e) {
  std::cerr << "FAIL: " << e.what() << '\n';
  return 1;
}
