#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include <cutlass/util/command_line.h>
#include <cutlass/util/device_memory.h>
#include "kernel/default_nb_atten.h"
#include "device/nb_atten.h"

namespace {
using namespace tiny_cutlass::natten;
using Factory = DefaultNBAtten<cutlass::arch::Sm80, cutlass::half_t,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 64, 16>,
    cutlass::gemm::GemmShape<32, 64, 16>>;
using Device = NBAtten<typename Factory::Kernel>;
using Element = Factory::Element;

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void check(cutlass::Status status) {
  if (status != cutlass::Status::kSuccess)
    throw std::runtime_error(cutlassGetStatusString(status));
}

struct Stream {
  cudaStream_t value{};
  Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(value); }
};
struct Event {
  cudaEvent_t value{};
  Event() { check(cudaEventCreate(&value)); }
  ~Event() { cudaEventDestroy(value); }
};

// Independent scalar reference: form the shifted window and enumerate its keys,
// then evaluate softmax in double precision using already-rounded FP16 inputs.
// This intentionally does not call the device window helper or tile metadata.
void reference(NeighborhoodProblem const& p, std::vector<Element> const& q,
    std::vector<Element> const& k, std::vector<Element> const& v,
    std::vector<double>& out, std::vector<double>& lse) {
  int lse_pitch = (p.length + 31) / 32 * 32;
  out.resize(size_t(p.batch_size) * p.length * p.heads * p.head_dim_value);
  lse.resize(Device::get_lse_size(p), 0.0);
  for (int b = 0; b < p.batch_size; ++b) {
    for (int h = 0; h < p.heads; ++h) {
      for (int x = 0; x < p.length; ++x) {
        int left = x - p.kernel_size / 2;
        int right = left + p.kernel_size;
        if (left < 0) { right -= left; left = 0; }
        if (right > p.length) { left -= right - p.length; right = p.length; }
        std::vector<double> scores;
        double peak = -std::numeric_limits<double>::infinity();
        size_t row = (size_t(b) * p.length + x) * p.heads + h;
        for (int y = left; y < right; ++y) {
          size_t key_row = (size_t(b) * p.length + y) * p.heads + h;
          double score = 0;
          for (int d = 0; d < p.head_dim; ++d)
            score += double(float(q[row * p.head_dim + d])) *
                     double(float(k[key_row * p.head_dim + d]));
          score *= p.scale;
          scores.push_back(score);
          peak = std::max(peak, score);
        }
        double sum = 0;
        for (double& score : scores) { score = std::exp(score - peak); sum += score; }
        lse[(size_t(b) * p.heads + h) * lse_pitch + x] = peak + std::log(sum);
        for (int d = 0; d < p.head_dim_value; ++d) {
          double result = 0;
          for (int y = left; y < right; ++y) {
            size_t key_row = (size_t(b) * p.length + y) * p.heads + h;
            result += scores[y - left] / sum * float(v[key_row * p.head_dim_value + d]);
          }
          out[row * p.head_dim_value + d] = result;
        }
      }
    }
  }
}

bool reject_contracts(Device::Arguments const& good) {
  auto fails = [](Device::Arguments const& a) {
    return Device::can_implement(a) != cutlass::Status::kSuccess;
  };
  auto a = good;
  a.problem.dilation = 2;
  if (!fails(a)) return false;
  a = good; a.problem.stride = 2;
  if (!fails(a)) return false;
  a = good; a.problem.head_dim = 7;
  if (!fails(a)) return false;
  a = good; a.problem.kernel_size = a.problem.length + 1;
  if (!fails(a)) return false;
  a = good; a.problem.length = 0;
  if (!fails(a)) return false;
  a = good; a.problem.scale = std::numeric_limits<float>::quiet_NaN();
  if (!fails(a)) return false;
  a = good; a.problem.head_dim_value = 72;
  if (!fails(a)) return false;
  a = good; a.query++;
  if (!fails(a)) return false;
  a = good; a.output = nullptr;
  if (!fails(a)) return false;
  Device uninitialized;
  return uninitialized.run() == cutlass::Status::kErrorInvalidProblem;
}

bool run_case(NeighborhoodProblem p, int mode, int repeats = 0) {
  size_t input_size = size_t(p.batch_size) * p.length * p.heads * p.head_dim;
  size_t output_size = size_t(p.batch_size) * p.length * p.heads * p.head_dim_value;
  if (p.length <= 0) throw std::runtime_error("Unsupported problem");
  std::mt19937 rng(20260918);
  std::uniform_real_distribution<float> random(-1.0f, 1.0f);
  std::vector<Element> q(input_size), k(input_size), v(output_size);
  for (auto& x : q) x = Element(mode == 1 ? 0.0f : random(rng) * (mode == 2 ? 8 : 1));
  for (auto& x : k) x = Element(random(rng));
  for (size_t i = 0; i < v.size(); ++i)
    v[i] = Element(mode == 1 ? float((i / (p.heads * p.head_dim_value)) % p.length) : random(rng));
  cutlass::device_memory::allocation<Element> dq(q.size()), dk(k.size()), dv(v.size());
  cutlass::device_memory::allocation<Element> dout(output_size + 16);
  cutlass::device_memory::allocation<float> dlse(Device::get_lse_size(p) + 16);
  Stream stream;
  check(cudaMemcpyAsync(dq.get(), q.data(), q.size() * sizeof(Element), cudaMemcpyHostToDevice, stream.value));
  check(cudaMemcpyAsync(dk.get(), k.data(), k.size() * sizeof(Element), cudaMemcpyHostToDevice, stream.value));
  check(cudaMemcpyAsync(dv.get(), v.data(), v.size() * sizeof(Element), cudaMemcpyHostToDevice, stream.value));
  check(cudaMemsetAsync(dout.get(), 0x7f, (output_size + 16) * sizeof(Element), stream.value));
  check(cudaMemsetAsync(dlse.get(), 0x7f, (Device::get_lse_size(p) + 16) * sizeof(float), stream.value));
  Device::Arguments args{dq.get(), dk.get(), dv.get(), dout.get(), dlse.get(), p};
  if (!reject_contracts(args)) throw std::runtime_error("Unsupported contract accepted");
  Device op;
  check(op.initialize(args));
  check(op.run(stream.value));
  std::vector<Element> actual(output_size + 16);
  std::vector<float> actual_lse(Device::get_lse_size(p) + 16);
  check(cudaMemcpyAsync(actual.data(), dout.get(), actual.size() * sizeof(Element), cudaMemcpyDeviceToHost, stream.value));
  check(cudaMemcpyAsync(actual_lse.data(), dlse.get(), actual_lse.size() * sizeof(float), cudaMemcpyDeviceToHost, stream.value));
  check(cudaStreamSynchronize(stream.value));
  std::vector<double> expected, expected_lse;
  reference(p, q, k, v, expected, expected_lse);
  double mae = 0, max_error = 0, lse_error = 0;
  for (size_t i = 0; i < output_size; ++i) {
    double error = std::abs(float(actual[i]) - expected[i]);
    if (!std::isfinite(error)) throw std::runtime_error("Non-finite output");
    mae += error;
    max_error = std::max(max_error, error);
  }
  mae /= output_size;
  int pitch = (p.length + 31) / 32 * 32;
  for (int bh = 0; bh < p.batch_size * p.heads; ++bh) {
    for (int x = 0; x < p.length; ++x) {
      size_t i = size_t(bh) * pitch + x;
      double error = std::abs(actual_lse[i] - expected_lse[i]);
      if (!std::isfinite(error)) throw std::runtime_error("Non-finite LSE");
      lse_error = std::max(lse_error, error);
    }
  }
  auto* out_bytes = reinterpret_cast<uint8_t*>(actual.data() + output_size);
  auto* lse_bytes = reinterpret_cast<uint8_t*>(actual_lse.data() + Device::get_lse_size(p));
  for (int i = 0; i < 16 * sizeof(Element); ++i)
    if (out_bytes[i] != 0x7f) throw std::runtime_error("Output guard overwritten");
  for (int i = 0; i < 16 * sizeof(float); ++i)
    if (lse_bytes[i] != 0x7f) throw std::runtime_error("LSE guard overwritten");
  bool passed = mae <= 1e-3 && max_error <= 1e-2 && lse_error <= 1e-3;
  std::cout << std::setprecision(8) << "B=" << p.batch_size << " L=" << p.length
      << " H=" << p.heads << " D=" << p.head_dim << " Dv=" << p.head_dim_value
      << " W=" << p.kernel_size << " mode=" << mode << " mae=" << mae
      << " max=" << max_error << " lse_max=" << lse_error
      << " " << (passed ? "PASS" : "FAIL") << '\n';
  if (!passed) return false;
  // The optional-LSE route must also preserve the output.
  args.logsumexp = nullptr;
  check(op.initialize(args));
  check(op.run(stream.value));
  check(cudaMemcpyAsync(actual.data(), dout.get(), output_size * sizeof(Element), cudaMemcpyDeviceToHost, stream.value));
  check(cudaStreamSynchronize(stream.value));
  for (size_t i = 0; i < output_size; ++i)
    if (!std::isfinite(float(actual[i])) || std::abs(float(actual[i]) - expected[i]) > 1e-2)
      throw std::runtime_error("Optional LSE path mismatch");
  if (repeats) {
    args.logsumexp = dlse.get();
    check(op.initialize(args));
    for (int i = 0; i < 10; ++i) check(op.run(stream.value));
    Event begin, end;
    check(cudaEventRecord(begin.value, stream.value));
    for (int i = 0; i < repeats; ++i) check(op.run(stream.value));
    check(cudaEventRecord(end.value, stream.value));
    check(cudaEventSynchronize(end.value));
    float elapsed = 0;
    check(cudaEventElapsedTime(&elapsed, begin.value, end.value));
    std::cout << "BENCH verified=true warmup=10 repeats=" << repeats
        << " forward_us=" << elapsed * 1000 / repeats << '\n';
  }
  return true;
}

NeighborhoodProblem problem(int b, int l, int h, int d, int dv, int w, float scale = -1) {
  NeighborhoodProblem p;
  p.batch_size = b; p.length = l; p.heads = h;
  p.head_dim = d; p.head_dim_value = dv; p.kernel_size = w;
  p.scale = scale < 0 ? 1.0f / std::sqrt(float(d)) : scale;
  return p;
}
} // namespace

int main(int argc, char const** argv) {
  try {
    cutlass::CommandLine cmd(argc, argv);
    if (cmd.check_cmd_line_flag("help")) {
      std::cout << "natten_fna --suite | [--length=1024 --kernel_size=33 --batch_size=2 "
          "--heads=4 --head_dim=64 --head_dim_value=64 --bench --iterations=50]\n";
      return 0;
    }
    int device = 0, driver = 0, runtime = 0;
    cudaDeviceProp prop{};
    check(cudaGetDevice(&device)); check(cudaGetDeviceProperties(&prop, device));
    check(cudaDriverGetVersion(&driver)); check(cudaRuntimeGetVersion(&runtime));
    std::cout << "GPU=" << prop.name << " CC=" << prop.major << prop.minor
        << " runtime=" << runtime << " driver=" << driver << '\n';
    if (cmd.check_cmd_line_flag("suite")) {
      struct Case { int b, l, h, d, dv, w, mode; };
      Case cases[] = {{1,1,1,8,8,1,0}, {1,8,1,8,8,3,1},
          {1,8,1,8,8,4,1}, {1,31,2,16,32,7,0}, {2,64,3,32,64,33,0},
          {2,65,2,64,32,1,0}, {2,65,3,32,64,64,0}, {2,127,2,64,64,65,0},
          {1,129,3,64,16,97,0}, {2,257,2,64,64,257,0}, {1,513,2,32,64,33,0},
          {1,129,2,64,64,65,2}};
      for (auto c : cases)
        if (!run_case(problem(c.b,c.l,c.h,c.d,c.dv,c.w),c.mode)) return 1;
      if (!run_case(problem(1,65,2,32,32,33,0),0)) return 1;
      std::cout << "All 13 GPU/reference cases and rejection checks passed.\n";
      return 0;
    }
    int b=2,l=1024,h=4,d=64,dv=64,w=33,repeats=50;
    cmd.get_cmd_line_argument("batch_size",b); cmd.get_cmd_line_argument("length",l);
    cmd.get_cmd_line_argument("heads",h); cmd.get_cmd_line_argument("head_dim",d);
    cmd.get_cmd_line_argument("head_dim_value",dv); cmd.get_cmd_line_argument("kernel_size",w);
    cmd.get_cmd_line_argument("iterations",repeats);
    if (repeats <= 0) throw std::runtime_error("iterations must be positive");
    return run_case(problem(b,l,h,d,dv,w),0,cmd.check_cmd_line_flag("bench")?repeats:0) ? 0 : 1;
  } catch (std::exception const& e) {
    std::cerr << "ERROR: " << e.what() << '\n';
    return 1;
  }
}
