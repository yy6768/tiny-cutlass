#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "../../flash-attention/flash_attention.h"
#include "cudnn_reference.h"

namespace {

void checked(cudaError_t status, char const* action) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
}

struct Stream {
  cudaStream_t handle = nullptr;
  Stream() { checked(cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking), "Create stream"); }
  ~Stream() { if (handle) cudaStreamDestroy(handle); }
};

struct Options {
  Problem problem;
  int seed = 3080;
  int iterations = 20;
  float input_scale = 1.f;
  double mae_tolerance = 1e-6;
  double max_abs_tolerance = 1e-2;
  bool verify_only = true;
  bool verify_reuse = true;
  bool help = false;
  std::string dump_prefix;
  bool diagnostic = false;
  std::string diagnostic_state;

  void parse(int argc, char const** argv) {
    cutlass::CommandLine cmd(argc, argv);
    help = cmd.check_cmd_line_flag("help");
    cmd.get_cmd_line_argument("batch_size", problem.batch_size, 16);
    cmd.get_cmd_line_argument("head_number", problem.head_number, 12);
    cmd.get_cmd_line_argument("seq_length", problem.seq_length, 1024);
    cmd.get_cmd_line_argument("seq_length_kv", problem.seq_length_kv, problem.seq_length);
    cmd.get_cmd_line_argument("head_size", problem.head_size, 64);
    cmd.get_cmd_line_argument("head_size_v", problem.head_size_v, problem.head_size);
    cmd.get_cmd_line_argument("seed", seed, 3080);
    cmd.get_cmd_line_argument("input-scale", input_scale, 1.f);
    cmd.get_cmd_line_argument("mae-tolerance", mae_tolerance, 1e-6);
    cmd.get_cmd_line_argument("max-abs-tolerance", max_abs_tolerance, 1e-2);
    cmd.get_cmd_line_argument("iterations", iterations, 20);
    cmd.get_cmd_line_argument("verify-only", verify_only, true);
    cmd.get_cmd_line_argument("verify-reuse", verify_reuse, true);
    cmd.get_cmd_line_argument("dump-prefix", dump_prefix, std::string());
    diagnostic = cmd.check_cmd_line_flag("diagnostic-state");
    cmd.get_cmd_line_argument("diagnostic-state", diagnostic_state, std::string());
    std::string kernel, reference;
    bool reference_check = true;
    cmd.get_cmd_line_argument("kernel", kernel, std::string("03-split-q"));
    cmd.get_cmd_line_argument("reference", reference, std::string("cudnn"));
    cmd.get_cmd_line_argument("reference-check", reference_check, true);
    if (help) return;
    if (diagnostic && diagnostic_state != "own" && diagnostic_state != "reference-lse" &&
        diagnostic_state != "reference-output" && diagnostic_state != "reference-both")
      throw std::runtime_error("diagnostic-state must be own, reference-lse, reference-output or reference-both.");
    if (kernel != "03-split-q" || reference != "cudnn" || !reference_check ||
        problem.batch_size <= 0 || problem.head_number <= 0 || problem.seq_length <= 0 ||
        problem.seq_length_kv <= 0 || problem.head_size <= 0 || problem.head_size_v <= 0 ||
        iterations <= 0 || !std::isfinite(input_scale) || input_scale < 0.f ||
        !std::isfinite(mae_tolerance) || mae_tolerance <= 0. || mae_tolerance > 1e-6 ||
        !std::isfinite(max_abs_tolerance) || max_abs_tolerance <= 0. || max_abs_tolerance > 1e-2)
      throw std::runtime_error("Invalid options; backward requires cuDNN, MAE < 1e-6, max abs <= 1e-2.");
    cmd.get_cmd_line_argument("attention-scale", problem.scale,
                              1.f / std::sqrt(float(problem.head_size)));
    if (!std::isfinite(problem.scale))
      throw std::runtime_error("Attention scale must be finite.");
  }
};

void random_input(cutlass::DeviceAllocation<Element>& allocation, int64_t count,
                  int seed, float scale = 1.f) {
  std::vector<Element> values(count);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> uniform(-2.f, 2.f);
  for (auto& value : values) value = Element(uniform(rng) * scale);
  cutlass::device_memory::copy_to_device(allocation.get(), values.data(), count);
}

struct Metrics {
  double mae = 0.;
  double max_abs = 0.;
  bool finite = true;
  bool passed = false;
  int64_t max_index = 0;
  double actual_at_max = 0.;
  double reference_at_max = 0.;
};

template <class Value>
Metrics compare(Value const* actual, Value const* expected, int64_t count,
                Options const& options, bool zero_expected = false, bool exact = false) {
  std::vector<Value> a(count), b(count);
  cutlass::device_memory::copy_to_host(a.data(), actual, count);
  if (!zero_expected) cutlass::device_memory::copy_to_host(b.data(), expected, count);
  Metrics result;
  long double sum = 0.;
  for (int64_t i = 0; i < count; ++i) {
    double av = double(float(a[i]));
    double bv = zero_expected ? 0. : double(float(b[i]));
    if (!std::isfinite(av) || !std::isfinite(bv)) {
      result.finite = false;
      result.mae = result.max_abs = std::numeric_limits<double>::infinity();
      return result;
    }
    double error = std::abs(av - bv);
    sum += error;
    if (error > result.max_abs) {
      result.max_abs = error;
      result.max_index = i;
      result.actual_at_max = av;
      result.reference_at_max = bv;
    }
  }
  result.mae = double(sum / count);
  result.passed = exact ? result.max_abs == 0. :
      result.mae < options.mae_tolerance && result.max_abs <= options.max_abs_tolerance;
  return result;
}

void print_metrics(char const* phase, char const* name, Metrics const& value) {
  std::cout << phase << " " << name << ": MAE=" << value.mae
            << " max_abs=" << value.max_abs << " finite=" << value.finite
            << " passed=" << value.passed << " max_index=" << value.max_index
            << " actual=" << value.actual_at_max << " reference=" << value.reference_at_max << "\n";
}

template <class Value>
void dump_tensor(std::string const& prefix, char const* name, Value const* data, int64_t count) {
  if (prefix.empty()) return;
  namespace fs = std::filesystem;
  fs::path source = fs::absolute(fs::path(__FILE__));
  fs::path build_root = fs::weakly_canonical(source.parent_path().parent_path().parent_path().parent_path() / "build");
  fs::path output = fs::weakly_canonical(fs::path(prefix + "-" + name + ".bin"));
  auto root_part = build_root.begin(), output_part = output.begin();
  for (; root_part != build_root.end() && output_part != output.end(); ++root_part, ++output_part)
    if (*root_part != *output_part) throw std::runtime_error("Dump prefix must stay under repository build/.");
  if (root_part != build_root.end()) throw std::runtime_error("Dump prefix must stay under repository build/.");
  fs::create_directories(output.parent_path());
  std::vector<Value> host(count);
  cutlass::device_memory::copy_to_host(host.data(), data, count);
  std::ofstream file(output, std::ios::binary);
  file.write(reinterpret_cast<char const*>(host.data()), std::streamsize(count * sizeof(Value)));
  if (!file) throw std::runtime_error("Cannot write diagnostic tensor dump.");
}

void poison_gradients(BackwardTensors const& t, Problem const& p, cudaStream_t stream) {
  checked(cudaMemsetAsync(t.grad_query, 0x7f, total_query_elements(p) * sizeof(Element), stream), "Poison dQ");
  checked(cudaMemsetAsync(t.grad_key, 0x7f, total_key_elements(p) * sizeof(Element), stream), "Poison dK");
  checked(cudaMemsetAsync(t.grad_value, 0x7f, total_value_elements(p) * sizeof(Element), stream), "Poison dV");
}

bool compare_gradients(BackwardTensors const& actual, BackwardTensors const& reference,
                       Problem const& p, Options const& options, bool zero = false) {
  Metrics dq = compare(actual.grad_query, reference.grad_query, total_query_elements(p), options, zero, zero);
  Metrics dk = compare(actual.grad_key, reference.grad_key, total_key_elements(p), options, zero, zero);
  Metrics dv = compare(actual.grad_value, reference.grad_value, total_value_elements(p), options, zero, zero);
  char const* phase = zero ? "Zero-dO" : "Gradient";
  print_metrics(phase, "dQ", dq);
  print_metrics(phase, "dK", dk);
  print_metrics(phase, "dV", dv);
  return dq.passed && dk.passed && dv.passed;
}

int run(Options const& options) {
  Problem const& p = options.problem;
  if (options.diagnostic)
    std::cout << "DIAGNOSTIC state=" << options.diagnostic_state
              << " (not an acceptance run; no pass stamp, reuse or timing)\n";
  auto const& forward = kernel_03_split_q();
  auto const& backward = backward_03_split_q();
  std::string reason;
  if (!forward.can_run(p, reason) || !backward.can_run(p, reason))
    throw std::runtime_error("Unsupported backward problem: " + reason);
  int64_t nq = total_query_elements(p), nk = total_key_elements(p);
  int64_t nv = total_value_elements(p), no = total_output_elements(p);
  int64_t nl = int64_t(p.batch_size) * p.head_number * p.seq_length;
  cutlass::DeviceAllocation<Element> q(nq), k(nk), v(nv), o(no), dout(no);
  cutlass::DeviceAllocation<Element> reference_o(no), dq(nq), dk(nk), dv(nv);
  cutlass::DeviceAllocation<Element> reference_dq(nq), reference_dk(nk), reference_dv(nv);
  cutlass::DeviceAllocation<float> lse(nl), reference_lse(nl);
  std::size_t forward_bytes = forward.workspace_bytes(p), backward_bytes = backward.workspace_bytes(p);
  cutlass::DeviceAllocation<uint8_t> forward_storage(forward_bytes), backward_storage(backward_bytes);
  Workspace forward_workspace{forward_storage.get(), forward_bytes};
  Workspace backward_workspace{backward_storage.get(), backward_bytes};
  random_input(q, nq, options.seed + 1, options.input_scale);
  random_input(k, nk, options.seed + 2, options.input_scale);
  random_input(v, nv, options.seed + 3);
  random_input(dout, no, options.seed + 4);
  Stream stream;
  Tensors forward_tensors{q.get(), k.get(), v.get(), o.get(), lse.get()};
  Tensors reference_forward{q.get(), k.get(), v.get(), reference_o.get(), reference_lse.get()};
  BackwardTensors tensors{q.get(), k.get(), v.get(), o.get(), dout.get(), lse.get(),
                           dq.get(), dk.get(), dv.get()};
  BackwardTensors reference{q.get(), k.get(), v.get(), reference_o.get(), dout.get(), reference_lse.get(),
                             reference_dq.get(), reference_dk.get(), reference_dv.get()};
  checked(cudaMemsetAsync(o.get(), 0x7f, no * sizeof(Element), stream.handle), "Poison O");
  checked(cudaMemsetAsync(reference_o.get(), 0x7f, no * sizeof(Element), stream.handle), "Poison reference O");
  checked(cudaMemsetAsync(lse.get(), 0xff, nl * sizeof(float), stream.handle), "Poison LSE");
  checked(cudaMemsetAsync(reference_lse.get(), 0xff, nl * sizeof(float), stream.handle), "Poison reference LSE");
  checked(forward.run(p, forward_tensors, forward_workspace, stream.handle), "Candidate training forward");
  std::string reference_error;
  cudaError_t status = run_cudnn_reference(p, reference_forward, stream.handle, reference_error);
  if (status != cudaSuccess) throw std::runtime_error("cuDNN training forward: " + reference_error);
  checked(cudaStreamSynchronize(stream.handle), "Synchronize training forward");
  Metrics output_metrics = compare(o.get(), reference_o.get(), no, options);
  print_metrics("Training", "O", output_metrics);
  Metrics stats_metrics = compare(lse.get(), reference_lse.get(), nl, options);
  // LSE is auxiliary FP32 state. Record its error; gradient parity is the gate.
  stats_metrics.passed = stats_metrics.finite;
  print_metrics("Training", "LSE", stats_metrics);
  dump_tensor(options.dump_prefix, "q-half", q.get(), nq);
  dump_tensor(options.dump_prefix, "k-half", k.get(), nk);
  dump_tensor(options.dump_prefix, "v-half", v.get(), nv);
  dump_tensor(options.dump_prefix, "do-half", dout.get(), no);
  dump_tensor(options.dump_prefix, "o-half", o.get(), no);
  dump_tensor(options.dump_prefix, "o-reference-half", reference_o.get(), no);
  dump_tensor(options.dump_prefix, "lse-float", lse.get(), nl);
  dump_tensor(options.dump_prefix, "lse-reference-float", reference_lse.get(), nl);
  if ((!output_metrics.passed || !stats_metrics.finite) && !options.diagnostic)
    throw std::runtime_error("Training forward parity/nonfinite LSE failed; backward not run.");

  // These substitutions only diagnose saved-state rounding. cuDNN always uses
  // its own independently computed training state. The diagnostic path returns
  // before normal acceptance, reuse and timing, even when the metrics pass.
  if (options.diagnostic_state == "reference-lse" || options.diagnostic_state == "reference-both")
    tensors.logsumexp = reference.logsumexp;
  if (options.diagnostic_state == "reference-output" || options.diagnostic_state == "reference-both")
    tensors.output = reference.output;

  poison_gradients(tensors, p, stream.handle);
  poison_gradients(reference, p, stream.handle);
  if (backward_bytes) {
    if (backward.run(p, tensors, {nullptr, backward_bytes}, stream.handle) != cudaErrorInvalidValue ||
        backward.run(p, tensors, {backward_workspace.data, backward_bytes - 1}, stream.handle) != cudaErrorInvalidValue)
      throw std::runtime_error("Expected rejection of missing/undersized backward workspace.");
    std::cout << "Workspace rejection passed: missing and undersized\n";
    checked(cudaMemsetAsync(backward_workspace.data, 0xff, backward_bytes, stream.handle), "Poison workspace");
  } else {
    std::cout << "Workspace rejection: not applicable (workspace_bytes=0)\n";
  }
  checked(backward.run(p, tensors, backward_workspace, stream.handle), "Candidate backward");
  bool singleton = p.seq_length == 1 && p.seq_length_kv == 1;
  if (singleton) {
    // The cuDNN SDPA backward frontend explicitly rejects Sq=Sk=1. Keep this
    // input as a separately labelled exact identity, not a cuDNN parity claim.
    checked(cudaMemsetAsync(reference.grad_query, 0, nq * sizeof(Element), stream.handle), "Identity dQ");
    checked(cudaMemsetAsync(reference.grad_key, 0, nk * sizeof(Element), stream.handle), "Identity dK");
    checked(cudaMemcpyAsync(reference.grad_value, dout.get(), nv * sizeof(Element),
                            cudaMemcpyDeviceToDevice, stream.handle), "Identity dV");
    std::cout << "Backward reference: analytic singleton (dQ=dK=0,dV=dO; cuDNN SDPA backward unsupported)\n";
  } else {
    status = run_cudnn_backward_reference(p, reference, stream.handle, reference_error);
    if (status != cudaSuccess) throw std::runtime_error("cuDNN backward: " + reference_error);
    std::cout << "Backward reference: cuDNN SDPA backward (deterministic=true; independent training O/LSE)\n";
  }
  checked(cudaStreamSynchronize(stream.handle), "Synchronize backward");
  dump_tensor(options.dump_prefix, "dq-half", dq.get(), nq);
  dump_tensor(options.dump_prefix, "dk-half", dk.get(), nk);
  dump_tensor(options.dump_prefix, "dv-half", dv.get(), nv);
  dump_tensor(options.dump_prefix, "dq-reference-half", reference_dq.get(), nq);
  dump_tensor(options.dump_prefix, "dk-reference-half", reference_dk.get(), nk);
  dump_tensor(options.dump_prefix, "dv-reference-half", reference_dv.get(), nv);
  bool gradients_passed = compare_gradients(tensors, reference, p, options);
  if (options.diagnostic) {
    std::cout << "DIAGNOSTIC completed: state=" << options.diagnostic_state
              << " gradient_thresholds_met=" << gradients_passed
              << " (not end-to-end acceptance; no timing)\n";
    return 0;
  }
  if (!gradients_passed)
    throw std::runtime_error("Backward gradient parity failed.");

  if (options.verify_reuse) {
    checked(cudaMemsetAsync(dout.get(), 0, no * sizeof(Element), stream.handle), "Zero upstream gradient");
    poison_gradients(tensors, p, stream.handle);
    if (backward_bytes)
      checked(cudaMemsetAsync(backward_workspace.data, 0xff, backward_bytes, stream.handle), "Repoison workspace");
    checked(backward.run(p, tensors, backward_workspace, stream.handle), "Repeated backward");
    checked(cudaStreamSynchronize(stream.handle), "Synchronize repeated backward");
    if (!compare_gradients(tensors, reference, p, options, true))
      throw std::runtime_error("Zero-dO repeated invocation failed.");
    std::cout << "Backward reuse passed: workspace_bytes=" << backward_bytes << " (zero-dO; non-default stream)\n";
  }
  std::cout << "Backward reference passed: 03-split-q (end-to-end; non-default stream)\n";
  if (options.verify_only) return 0;

  // Restore the verified nonzero dO distribution before measuring backward.
  checked(cudaStreamSynchronize(stream.handle), "Synchronize before restore");
  random_input(dout, no, options.seed + 4);
  checked(backward.run(p, tensors, backward_workspace, stream.handle), "Backward warmup");
  checked(cudaStreamSynchronize(stream.handle), "Synchronize backward warmup");
  cudaEvent_t start = nullptr, stop = nullptr;
  checked(cudaEventCreate(&start), "Create start event");
  checked(cudaEventCreate(&stop), "Create stop event");
  checked(cudaEventRecord(start, stream.handle), "Record start event");
  for (int i = 0; i < options.iterations; ++i)
    checked(backward.run(p, tensors, backward_workspace, stream.handle), "Timed backward launch");
  checked(cudaEventRecord(stop, stream.handle), "Record stop event");
  checked(cudaEventSynchronize(stop), "Synchronize stop event");
  float elapsed_ms = 0.f;
  checked(cudaEventElapsedTime(&elapsed_ms, start, stop), "Elapsed time");
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.f)
    throw std::runtime_error("Invalid backward elapsed time.");
  std::cout << "Backward runtime: " << elapsed_ms / options.iterations << " ms\n";
  return 0;
}

} // namespace

int main(int argc, char const** argv) {
  try {
    std::cout << std::setprecision(12);
    Options options;
    options.parse(argc, argv);
    if (options.help) {
      std::cout << "flash_attention_backward --kernel=03-split-q\n"
                << "--batch_size --head_number --seq_length --seq_length_kv --head_size --head_size_v\n"
                << "--seed=3080 --input-scale=1 --verify-only=true --verify-reuse=true\n"
                << "--attention-scale=<finite float> (default: 1/sqrt(D))\n"
                << "--mae-tolerance=1e-6 --max-abs-tolerance=1e-2 --iterations=20\n"
                << "--dump-prefix=<path under build/> (optional raw diagnostic tensor files)\n"
                << "--diagnostic-state=own|reference-lse|reference-output|reference-both\n"
                << "Diagnostic mode cannot produce an acceptance stamp or benchmark.\n"
                << "End-to-end gradient verification precedes optional backward timing.\n";
      return 0;
    }
    int device = 0;
    checked(cudaGetDevice(&device), "Get device");
    cudaDeviceProp props{};
    checked(cudaGetDeviceProperties(&props, device), "Get device properties");
    std::cout << "Device: " << props.name << " (SM" << props.major << props.minor << ")\n";
    std::cout << "cuDNN version: " << cudnnGetVersion() << "\n";
    return run(options);
  } catch (std::exception const& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
