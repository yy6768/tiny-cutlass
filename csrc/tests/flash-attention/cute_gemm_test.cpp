/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

/*! \file
    \brief Shared harness for the CuTe warm-up GEMM.

    Owns input generation, the cuBLAS reference, MAE checking and timing, so
    csrc/flash-attention/cute-gemm/ only has to own the launch path.

    Reference is cuBLAS rather than cuDNN: it ships with the CUDA Toolkit, so
    there is no optional dependency and no fallback branch to write.
*/

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"

#include "../../flash-attention/cute-gemm/simple_gemm.h"

namespace {

struct Options {
  bool help = false;
  bool error = false;
  bool reference_check = true;

  int m = 4096;
  int n = 4096;
  int k = 4096;
  float alpha = 1.0f;
  float beta = 0.0f;

  int iterations = 20;
  int seed = 3080;
  float mae_tolerance = 1.0e-2f;

  void parse(int argc, char const** args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m, 4096);
    cmd.get_cmd_line_argument("n", n, 4096);
    cmd.get_cmd_line_argument("k", k, 4096);
    cmd.get_cmd_line_argument("alpha", alpha, 1.0f);
    cmd.get_cmd_line_argument("beta", beta, 0.0f);
    cmd.get_cmd_line_argument("iterations", iterations, 20);
    cmd.get_cmd_line_argument("seed", seed, 3080);
    cmd.get_cmd_line_argument("reference-check", reference_check, true);
    cmd.get_cmd_line_argument("mae-tolerance", mae_tolerance, 1.0e-2f);

    if (m <= 0 || n <= 0 || k <= 0 || iterations <= 0 || mae_tolerance <= 0.0f) {
      error = true;
    }
  }

  GemmProblem problem() const {
    GemmProblem p;
    p.m = m;
    p.n = n;
    p.k = k;
    p.alpha = alpha;
    p.beta = beta;
    return p;
  }

  // 2 * M * N * K fused multiply-adds.
  double gflops(double runtime_s) const {
    return 2.0 * double(m) * double(n) * double(k) / 1.0e9 / runtime_s;
  }

  std::ostream& print_usage(std::ostream& out) const {
    out << "cute_gemm_test\n\n"
        << "  C = alpha * A * B^T + beta * C   (A: MxK row-major, B: NxK row-major, C: MxN row-major)\n\n"
        << "Options:\n\n"
        << "  --help                    Display this usage statement.\n"
        << "  --m=<int>                 Rows of A / C (default: 4096).\n"
        << "  --n=<int>                 Rows of B / cols of C (default: 4096).\n"
        << "  --k=<int>                 Contraction extent (default: 4096).\n"
        << "  --alpha=<float>           Scales A*B^T (default: 1.0).\n"
        << "  --beta=<float>            Scales the incoming C (default: 0.0).\n"
        << "  --reference-check=<bool>  Verify against cuBLAS before timing (default: true).\n"
        << "  --mae-tolerance=<float>   Required MAE against cuBLAS (default: 1e-2).\n"
        << "  --iterations=<int>        Timed iterations (default: 20).\n"
        << "  --seed=<int>              Random seed base (default: 3080).\n";
    return out;
  }
};

struct CompareResult {
  double mae = 0.0;
  float max_abs = 0.0f;
  int64_t max_index = 0;
  bool passed = false;
};

void fill_random_uniform(
    cutlass::DeviceAllocation<GemmElement>& block,
    int64_t elements,
    int seed) {
  std::vector<GemmElement> host(elements);
  std::mt19937 rng(seed);
  // Narrow range keeps the FP16 accumulation error well inside the tolerance at
  // K = 4096, so a real MAE regression is not hidden by input noise.
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

  for (auto& value : host) {
    value = GemmElement(distribution(rng));
  }

  cutlass::device_memory::copy_to_device(block.get(), host.data(), elements);
}

CompareResult compare_outputs(
    cutlass::DeviceAllocation<GemmElement> const& output,
    cutlass::DeviceAllocation<GemmElement> const& reference,
    int64_t elements,
    float mae_tolerance) {
  std::vector<GemmElement> host_output(elements);
  std::vector<GemmElement> host_reference(elements);

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

// cuBLAS is column-major. Our tensors are row-major, and a row-major (R, C)
// buffer is bit-identical to a column-major (C, R) buffer, so:
//
//   row-major   C(M,N) = A(M,K) * B(N,K)^T
//   col-major   C'(N,M) = B'(K,N)^T * A'(K,M)
//
// where X' is the same bytes reinterpreted. That is one gemm call with
// transa = T (on B), transb = N (on A), m = N, n = M, k = K -- no explicit
// transposes and no extra buffers.
cudaError_t run_cublas_reference(
    cublasHandle_t handle,
    GemmProblem const& problem,
    GemmTensors const& tensors,
    std::string& error) {
  float alpha = problem.alpha;
  float beta = problem.beta;

  cublasStatus_t status = cublasGemmEx(
      handle,
      CUBLAS_OP_T,               // B' is (K,N) col-major, transpose it
      CUBLAS_OP_N,               // A' is (K,M) col-major, use as-is
      problem.n,                 // rows of the col-major output
      problem.m,                 // cols of the col-major output
      problem.k,
      &alpha,
      tensors.b, CUDA_R_16F, problem.k,   // ldb' = K
      tensors.a, CUDA_R_16F, problem.k,   // lda' = K
      &beta,
      tensors.c, CUDA_R_16F, problem.n,   // ldc' = N
      CUBLAS_COMPUTE_32F,                 // FP32 accumulate, matching our kernel
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  if (status != CUBLAS_STATUS_SUCCESS) {
    error = "cublasGemmEx failed with status " + std::to_string(int(status));
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

int run(Options const& options) {
  GemmProblem problem = options.problem();

  std::string unsupported_reason;
  if (!cute_simple_gemm_can_run(problem, unsupported_reason)) {
    std::cerr << "Kernel does not support this problem: " << unsupported_reason << "\n";
    return -1;
  }

  int64_t elements_a = int64_t(problem.m) * problem.k;
  int64_t elements_b = int64_t(problem.n) * problem.k;
  int64_t elements_c = int64_t(problem.m) * problem.n;

  cutlass::DeviceAllocation<GemmElement> block_a(elements_a);
  cutlass::DeviceAllocation<GemmElement> block_b(elements_b);
  cutlass::DeviceAllocation<GemmElement> block_c(elements_c);
  cutlass::DeviceAllocation<GemmElement> block_c_initial(elements_c);
  cutlass::DeviceAllocation<GemmElement> block_reference_c(elements_c);

  fill_random_uniform(block_a, elements_a, options.seed + 1);
  fill_random_uniform(block_b, elements_b, options.seed + 2);
  // beta != 0 reads C, so both paths must start from the same bytes.
  fill_random_uniform(block_c_initial, elements_c, options.seed + 3);

  auto reset_output = [&](cutlass::DeviceAllocation<GemmElement>& dst) {
    return cudaMemcpy(
        dst.get(), block_c_initial.get(),
        elements_c * sizeof(GemmElement), cudaMemcpyDeviceToDevice);
  };

  GemmTensors tensors;
  tensors.a = block_a.get();
  tensors.b = block_b.get();
  tensors.c = block_c.get();

  cudaError_t err = reset_output(block_c);
  if (err != cudaSuccess) {
    std::cerr << "Output init failed: " << cudaGetErrorString(err) << "\n";
    return -1;
  }

  err = cute_simple_gemm(problem, tensors, nullptr);
  if (err != cudaSuccess) {
    std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << "\n";
    return -1;
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "Kernel sync failed: " << cudaGetErrorString(err) << "\n";
    return -1;
  }

  cublasHandle_t handle = nullptr;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasCreate failed\n";
    return -1;
  }

  int status = 0;

  if (options.reference_check) {
    err = reset_output(block_reference_c);
    if (err != cudaSuccess) {
      std::cerr << "Reference init failed: " << cudaGetErrorString(err) << "\n";
      cublasDestroy(handle);
      return -1;
    }

    GemmTensors reference_tensors = tensors;
    reference_tensors.c = block_reference_c.get();

    std::string reference_error;
    err = run_cublas_reference(handle, problem, reference_tensors, reference_error);
    if (err != cudaSuccess) {
      std::cerr << "cuBLAS reference failed: " << reference_error << "\n";
      cublasDestroy(handle);
      return -1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::cerr << "cuBLAS reference sync failed: " << cudaGetErrorString(err) << "\n";
      cublasDestroy(handle);
      return -1;
    }

    CompareResult compare = compare_outputs(
        block_c, block_reference_c, elements_c, options.mae_tolerance);

    std::cout << "    Reference: cuBLAS GemmEx (FP32 accumulate)\n"
              << "    MAE      : " << compare.mae
              << " (tolerance " << options.mae_tolerance << ")\n"
              << "    Max abs  : " << compare.max_abs
              << " at index " << compare.max_index << "\n";

    if (!compare.passed) {
      std::cout << "\nFailed\n";
      cublasDestroy(handle);
      return -1;
    }
  }

  cublasDestroy(handle);

  // Warmup, then timed loop. beta is applied every iteration; with beta != 0
  // that means C compounds across iterations, which is fine for timing but is
  // why verification ran on a freshly reset buffer above.
  err = cute_simple_gemm(problem, tensors, nullptr);
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
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
    err = cute_simple_gemm(problem, tensors, nullptr);
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

  std::cout << "\n" << cute_simple_gemm_name() << ":\n"
            << "====================================================\n"
            << "    {M, N, K} = {" << options.m << ", " << options.n << ", "
            << options.k << "}\n"
            << "    alpha/beta: " << options.alpha << " / " << options.beta << "\n"
            << "    Runtime : " << runtime_ms << " ms\n"
            << "    TFLOPs  : " << options.gflops(runtime_ms / 1000.0) / 1000.0 << "\n"
            << "\nPassed\n";

  return status;
}

}  // namespace

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

  return run(options);
}
