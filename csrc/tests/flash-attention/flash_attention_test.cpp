#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"

#include "../../flash-attention/flash_attention.h"
#include "cudnn_reference.h"

#ifndef FLASH_ATTENTION_DEFAULT_KERNEL
#define FLASH_ATTENTION_DEFAULT_KERNEL "all"
#endif

namespace {

struct Options {
  bool help = false;
  bool error = false;
  bool reference_check = true;

  std::string kernel = FLASH_ATTENTION_DEFAULT_KERNEL;
  std::string reference = "cudnn";

  int head_number = 12;
  int batch_size = 16;
  int head_size = 64;
  int head_size_v = 64;
  int seq_length = 1024;
  int seq_length_kv = 1024;
  int iterations = 20;
  int seed = 3080;

  float mae_tolerance = 1.0e-3f;

  void parse(int argc, char const** args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("kernel", kernel, kernel);
    cmd.get_cmd_line_argument("reference", reference, reference);
    cmd.get_cmd_line_argument("reference-check", reference_check, true);
    cmd.get_cmd_line_argument("head_number", head_number, 12);
    cmd.get_cmd_line_argument("batch_size", batch_size, 16);
    cmd.get_cmd_line_argument("head_size", head_size, 64);
    cmd.get_cmd_line_argument("head_size_v", head_size_v, head_size);
    cmd.get_cmd_line_argument("seq_length", seq_length, 1024);
    cmd.get_cmd_line_argument("seq_length_kv", seq_length_kv, seq_length);
    cmd.get_cmd_line_argument("iterations", iterations, 20);
    cmd.get_cmd_line_argument("seed", seed, 3080);
    cmd.get_cmd_line_argument("mae-tolerance", mae_tolerance, 1.0e-3f);

    if (head_number <= 0 || batch_size <= 0 || head_size <= 0 ||
        head_size_v <= 0 || seq_length <= 0 || seq_length_kv <= 0 ||
        iterations <= 0 || mae_tolerance <= 0.0f) {
      error = true;
    }
  }

  Problem problem() const {
    Problem p;
    p.head_number = head_number;
    p.batch_size = batch_size;
    p.head_size = head_size;
    p.head_size_v = head_size_v;
    p.seq_length = seq_length;
    p.seq_length_kv = seq_length_kv;
    p.scale = 1.0f / std::sqrt(float(head_size));
    return p;
  }

  double gflops(double runtime_s) const {
    int64_t fops = int64_t(2) * batch_size * head_number * seq_length * seq_length_kv * head_size
                 + int64_t(2) * batch_size * head_number * seq_length * seq_length_kv * head_size_v
                 + int64_t(3) * batch_size * head_number * seq_length * seq_length_kv;
    return double(fops) / 1.0e9 / runtime_s;
  }

  std::ostream& print_usage(std::ostream& out) const {
    out << "flash_attention_test\n\n"
        << "Options:\n\n"
        << "  --help                         Display this usage statement.\n"
        << "  --kernel=<id|all|list>         Kernel to run. Available: 00-naive, 01-online-softmax, 02-tiled-online, all.\n"
        << "  --reference=<cudnn>            Reference backend. CPU reference is intentionally unsupported.\n"
        << "  --reference-check=<bool>       Run reference verification before timing (default: true).\n"
        << "  --mae-tolerance=<float>        Required MAE against reference (default: 1e-3).\n"
        << "  --head_number=<int>            Head number (default: 12).\n"
        << "  --batch_size=<int>             Batch size (default: 16).\n"
        << "  --head_size=<int>              Head dim for Q/K (default: 64).\n"
        << "  --head_size_v=<int>            Head dim for V/O (default: head_size).\n"
        << "  --seq_length=<int>             Sequence length for Q (default: 1024).\n"
        << "  --seq_length_kv=<int>          Sequence length for K/V (default: seq_length).\n"
        << "  --iterations=<int>             Number of timed iterations (default: 20).\n"
        << "  --seed=<int>                   Random seed base (default: 3080).\n";
    return out;
  }
};

struct CompareResult {
  double mae = 0.0;
  float max_abs = 0.0f;
  int64_t max_index = 0;
  bool passed = false;
};

Kernel const* const kKernels[] = {
    &kernel_00_naive(),
    &kernel_01_online_softmax(),
    &kernel_02_tiled_online(),
};

Kernel const* find_kernel(std::string const& id) {
  for (auto const* kernel : kKernels) {
    if (id == kernel->id) {
      return kernel;
    }
  }
  return nullptr;
}

void list_kernels(std::ostream& out) {
  for (auto const* kernel : kKernels) {
    out << "  " << kernel->id << " - " << kernel->name << "\n";
  }
}

CompareResult compare_outputs(
    cutlass::DeviceAllocation<Element> const& output,
    cutlass::DeviceAllocation<Element> const& reference,
    int64_t elements,
    float mae_tolerance) {
  std::vector<Element> host_output(elements);
  std::vector<Element> host_reference(elements);

  cutlass::device_memory::copy_to_host(host_output.data(), output.get(), elements);
  cutlass::device_memory::copy_to_host(host_reference.data(), reference.get(), elements);

  CompareResult result;
  long double abs_sum = 0.0;

  for (int64_t i = 0; i < elements; ++i) {
    float actual = float(host_output[i]);
    float expected = float(host_reference[i]);
    float diff = std::fabs(actual - expected);

    if (!std::isfinite(actual) || !std::isfinite(expected)) {
      result.max_index = i;
      result.max_abs = diff;
      result.mae = std::numeric_limits<double>::infinity();
      result.passed = false;
      return result;
    }

    abs_sum += diff;
    if (diff > result.max_abs) {
      result.max_abs = diff;
      result.max_index = i;
    }
  }

  result.mae = double(abs_sum / long double(elements));
  result.passed = result.mae <= double(mae_tolerance);
  return result;
}

void fill_random_uniform(
    cutlass::DeviceAllocation<Element>& block,
    int64_t elements,
    int seed) {
  std::vector<Element> host(elements);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> distribution(-2.0f, 2.0f);

  for (auto& value : host) {
    value = Element(distribution(rng));
  }

  cutlass::device_memory::copy_to_device(block.get(), host.data(), elements);
}

cudaError_t run_kernel_once(
    Kernel const& kernel,
    Problem const& problem,
    Tensors const& tensors,
    Workspace workspace) {
  cudaError_t err = kernel.run(problem, tensors, workspace, nullptr);
  if (err != cudaSuccess) {
    return err;
  }
  return cudaDeviceSynchronize();
}

int run_one(Kernel const& kernel, Options const& options) {
  Problem problem = options.problem();
  std::string unsupported_reason;
  if (kernel.can_run && !kernel.can_run(problem, unsupported_reason)) {
    std::cerr << "Kernel " << kernel.id << " does not support this problem: "
              << unsupported_reason << "\n";
    return -1;
  }

  if (options.reference_check && options.reference != "cudnn") {
    std::cerr << "Only --reference=cudnn is supported. CPU reference was intentionally removed.\n";
    return -1;
  }

  int64_t total_q = total_query_elements(problem);
  int64_t total_k = total_key_elements(problem);
  int64_t total_v = total_value_elements(problem);
  int64_t total_o = total_output_elements(problem);

  cutlass::DeviceAllocation<Element> block_q(total_q);
  cutlass::DeviceAllocation<Element> block_k(total_k);
  cutlass::DeviceAllocation<Element> block_v(total_v);
  cutlass::DeviceAllocation<Element> block_o(total_o);
  cutlass::DeviceAllocation<Element> block_reference_o(total_o);

  fill_random_uniform(block_q, total_q, options.seed + 1);
  fill_random_uniform(block_k, total_k, options.seed + 2);
  fill_random_uniform(block_v, total_v, options.seed + 3);
  cudaMemset(block_o.get(), 0, total_o * sizeof(Element));
  cudaMemset(block_reference_o.get(), 0, total_o * sizeof(Element));

  std::size_t workspace_bytes = kernel.workspace_bytes ? kernel.workspace_bytes(problem) : 0;
  cutlass::DeviceAllocation<uint8_t> block_workspace(workspace_bytes);

  Tensors tensors;
  tensors.query = block_q.get();
  tensors.key = block_k.get();
  tensors.value = block_v.get();
  tensors.output = block_o.get();

  Workspace workspace;
  workspace.data = block_workspace.get();
  workspace.bytes = workspace_bytes;

  cudaError_t err = run_kernel_once(kernel, problem, tensors, workspace);
  if (err != cudaSuccess) {
    std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << "\n";
    return -1;
  }

  if (options.reference_check) {
    Tensors reference_tensors = tensors;
    reference_tensors.output = block_reference_o.get();

    std::string reference_error;
    err = run_cudnn_reference(problem, reference_tensors, nullptr, reference_error);
    if (err != cudaSuccess) {
      std::cerr << "cuDNN reference failed: " << reference_error << "\n";
      return -1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::cerr << "cuDNN reference sync failed: " << cudaGetErrorString(err) << "\n";
      return -1;
    }

    CompareResult compare = compare_outputs(
        block_o, block_reference_o, total_o, options.mae_tolerance);
    std::cout << "    Reference: cuDNN SDPA\n"
              << "    MAE      : " << compare.mae << " (tolerance " << options.mae_tolerance << ")\n"
              << "    Max abs  : " << compare.max_abs << " at index " << compare.max_index << "\n";

    if (!compare.passed) {
      std::cout << "\nFailed\n";
      return -1;
    }
  }

  err = run_kernel_once(kernel, problem, tensors, workspace);
  if (err != cudaSuccess) {
    std::cerr << "Warmup failed: " << cudaGetErrorString(err) << "\n";
    return -1;
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < options.iterations; ++i) {
    err = kernel.run(problem, tensors, workspace, nullptr);
    if (err != cudaSuccess) {
      std::cerr << "Timed launch failed: " << cudaGetErrorString(err) << "\n";
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      return -1;
    }
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsed_ms = 0.0f;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double runtime_ms = double(elapsed_ms) / double(options.iterations);
  double gflops = options.gflops(runtime_ms / 1000.0);

  std::cout << "\n" << kernel.name << ":\n"
            << "====================================================\n"
            << "    {Sq, Sk, d, d_v, H, B} = {"
            << options.seq_length << ", " << options.seq_length_kv << ", "
            << options.head_size << ", " << options.head_size_v << ", "
            << options.head_number << ", " << options.batch_size << "}\n"
            << "    Runtime: " << runtime_ms << " ms\n"
            << "    GFLOPs : " << gflops << "\n"
            << "\nPassed\n";

  return 0;
}

} // namespace

int main(int argc, char const** args) {
  cudaDeviceProp props;
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties: " << cudaGetErrorString(error) << "\n";
    return -1;
  }

  std::cout << "Device: " << props.name << " (SM" << props.major << props.minor << ")\n";
  if (CUDART_VERSION < 11000 || props.major < 8) {
    std::cout << "This test requires Ampere (SM80) or later.\n";
    return 0;
  }

  Options options;
  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << "\n";
    return 0;
  }
  if (options.error) {
    std::cerr << "Invalid options.\n";
    return -1;
  }

  if (options.kernel == "list") {
    list_kernels(std::cout);
    return 0;
  }

  if (options.kernel == "all") {
    for (auto const* kernel : kKernels) {
      int status = run_one(*kernel, options);
      if (status != 0) {
        return status;
      }
    }
    return 0;
  }

  auto const* kernel = find_kernel(options.kernel);
  if (!kernel) {
    std::cerr << "Unknown kernel: " << options.kernel << "\nAvailable kernels:\n";
    list_kernels(std::cerr);
    return -1;
  }

  return run_one(*kernel, options);
}
