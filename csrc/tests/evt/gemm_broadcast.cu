/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*! \file
    \brief Verification + benchmark for the EVT residual-block GEMM (ported CUTLASS example 47).

        D = ((A.B + bias_row) + C1) + C2

    fused into ONE GEMM epilogue by an Sm80EVT tree. Two kernels are built from the same tree,
    differing only in threadblock swizzle:

      * data-parallel : GemmIdentityThreadblockSwizzle
      * StreamK       : ThreadblockSwizzleStreamK

    The reference is a host FP32 recomputation of the whole expression (matmul in FP32, then the
    three adds). Both kernels are checked against it, then timed, so the EVT epilogue cost is
    measured on a verified kernel.

    Usage:
      gemm_broadcast --verify              small shapes, correctness only
      gemm_broadcast --m=... --n=... --k=... --iterations=N
*/

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
// arch/mma.h defines the arch::Op* tags that reference/device/gemm.h uses as
// default template arguments; it does not include them itself.
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle_streamk.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/gemm.h"

#include "evt/gemm_broadcast/device/gemm_broadcast.h"
#include "evt/gemm_broadcast/kernel/default_gemm_broadcast.h"

#include "tests/common/test_utils.h"

namespace tiny_cutlass {
namespace evt {
namespace {

using Element = cutlass::half_t;

// Launches discarded before timing starts, to reach steady-state clocks.
constexpr int kWarmupIterations = 10;

// One policy per swizzle; everything else is shared. This is the whole point of
// keeping the swizzle a template parameter.
using DataParallelPolicy = kernel::DefaultGemmBroadcast<
    cutlass::arch::Sm80, Element, float, float,
    cutlass::layout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>;

using StreamKPolicy = kernel::DefaultGemmBroadcast<
    cutlass::arch::Sm80, Element, float, float,
    cutlass::layout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::gemm::threadblock::ThreadblockSwizzleStreamK>;

using DataParallelOp = device::GemmBroadcast<DataParallelPolicy>;
using StreamKOp = device::GemmBroadcast<StreamKPolicy>;

struct Options {
  bool help = false;
  bool error = false;

  int m = 2048;
  int n = 2048;
  int k = 1024;
  int split_k_slices = 1;
  int iterations = 100;
  float mae_tolerance = 2.0e-3f;

  void parse(int argc, char const** argv) {
    cutlass::CommandLine cmd(argc, argv);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    // --verify: small, fast, correctness-only shape.
    if (cmd.check_cmd_line_flag("verify")) {
      m = 512;
      n = 512;
      k = 256;
      iterations = 0;
    }

    cmd.get_cmd_line_argument("m", m, m);
    cmd.get_cmd_line_argument("n", n, n);
    cmd.get_cmd_line_argument("k", k, k);
    cmd.get_cmd_line_argument("split-k-slices", split_k_slices, split_k_slices);
    cmd.get_cmd_line_argument("iterations", iterations, iterations);
    cmd.get_cmd_line_argument("mae-tolerance", mae_tolerance, mae_tolerance);

    if (m <= 0 || n <= 0 || k <= 0) {
      error = true;
    }
  }

  static std::ostream& print_usage(std::ostream& out) {
    out << "gemm_broadcast : EVT-fused residual-block GEMM  D = ((A.B + bias) + C1) + C2\n\n"
        << "  --help                    print this message\n"
        << "  --verify                  small correctness-only run (512x512x256, no timing)\n"
        << "  --m=<int> --n=<int> --k=<int>   problem size (default 2048x2048x1024)\n"
        << "  --split-k-slices=<int>    split-K slices for the data-parallel kernel (default 1)\n"
        << "  --iterations=<int>        timing iterations (default 100, 0 disables)\n"
        << "  --mae-tolerance=<float>   MAE tolerance (default 2.0e-3)\n";
    return out;
  }

  double gflops(double runtime_s) const {
    // 2*M*N*K for the matmul plus 3*M*N for the three epilogue adds.
    double flops = 2.0 * double(m) * double(n) * double(k) + 3.0 * double(m) * double(n);
    return flops / runtime_s / 1.0e9;
  }
};

// ---------------------------------------------------------------------------
// Reference: a plain unfused CUTLASS device GEMM (FP32 accumulate) followed by
// a separate elementwise kernel for the three adds.
//
// Deliberately UNFUSED -- that is what makes it an independent check of the EVT
// epilogue: the reference does the bias broadcast and the two residual adds in
// their own kernel, reading intermediates back from global memory, exactly the
// work the visitor tree is supposed to have absorbed.
//
// This used to be a host triple loop. At bench shapes that is O(M*N*K) on the
// CPU (~10^11 ops at 4096x4096x2048): it ran for minutes, during which the GPU
// dropped to idle clocks, so the first kernel timed afterwards absorbed the
// entire clock ramp and read ~7x slower than NCU measured. Keep the reference on
// the device.
// ---------------------------------------------------------------------------
__global__ void apply_epilogue_reference(
    float const* __restrict__ acc,
    Element const* __restrict__ bias,
    Element const* __restrict__ c1,
    Element const* __restrict__ c2,
    Element* __restrict__ d,
    int m,
    int n) {
  int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = int64_t(m) * int64_t(n);
  if (index >= total) {
    return;
  }
  int col = int(index % int64_t(n));

  // Same association order as the visitor tree: ((acc + bias) + C1) + C2.
  float value = acc[index] + float(bias[col]);
  value += float(c1[index]);
  value += float(c2[index]);
  d[index] = Element(value);
}

/// Builds the reference D on the device. Returns false on any CUDA/CUTLASS error.
bool reference_device(
    Options const& options,
    Element const* d_a,
    Element const* d_b,
    Element const* d_bias,
    Element const* d_c1,
    Element const* d_c2,
    std::vector<Element>& reference) {
  int const M = options.m;
  int const N = options.n;
  int const K = options.k;

  testing::DeviceBuffer<float> d_acc(size_t(M) * size_t(N));
  if (d_acc.get() == nullptr) {
    return false;
  }

  // Unfused reference matmul: accumulate in FP32, no epilogue beyond alpha/beta.
  using RefGemm = cutlass::reference::device::Gemm<
      Element, cutlass::layout::RowMajor,
      Element, cutlass::layout::RowMajor,
      float, cutlass::layout::RowMajor,
      float, float>;

  // The reference Gemm's TensorRef is non-const in its element type.
  cutlass::TensorRef<Element, cutlass::layout::RowMajor> ref_a(const_cast<Element*>(d_a), K);
  cutlass::TensorRef<Element, cutlass::layout::RowMajor> ref_b(const_cast<Element*>(d_b), N);
  cutlass::TensorRef<float, cutlass::layout::RowMajor> ref_acc(d_acc.get(), N);

  RefGemm ref_gemm;
  ref_gemm(
      cutlass::gemm::GemmCoord(M, N, K),
      1.0f, ref_a, ref_b,
      0.0f, ref_acc, ref_acc,
      0.0f);

  cudaError_t error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    std::cerr << "reference GEMM failed: " << cudaGetErrorString(error) << "\n";
    return false;
  }

  testing::DeviceBuffer<Element> d_ref(size_t(M) * size_t(N));
  if (d_ref.get() == nullptr) {
    return false;
  }

  int64_t total = int64_t(M) * int64_t(N);
  int const threads = 256;
  int64_t blocks = (total + threads - 1) / threads;

  apply_epilogue_reference<<<dim3(unsigned(blocks)), dim3(threads)>>>(
      d_acc.get(), d_bias, d_c1, d_c2, d_ref.get(), M, N);

  error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    std::cerr << "reference epilogue failed: " << cudaGetErrorString(error) << "\n";
    return false;
  }

  reference.resize(size_t(total));
  return d_ref.copy_to_host(reference);
}

template <typename Op>
bool run_one(
    char const* description,
    Options const& options,
    typename Op::Arguments args,
    std::vector<Element> const& reference,
    double& avg_runtime_ms) {
  avg_runtime_ms = 0.0;

  size_t workspace_size = Op::get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  cutlass::Status status = Op::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "  " << description << ": can_implement failed ("
              << cutlass::cutlassGetStatusString(status) << ")\n";
    return false;
  }

  // Initialize ONCE, outside the timing loop. Params precomputation does device
  // queries and occupancy math; timing it would swamp the kernel itself.
  Op op;
  status = op.initialize(args, workspace.get(), nullptr);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "  " << description << ": initialize failed ("
              << cutlass::cutlassGetStatusString(status) << ")\n";
    return false;
  }

  status = op.run(nullptr);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "  " << description << ": run failed ("
              << cutlass::cutlassGetStatusString(status) << ")\n";
    return false;
  }

  cudaError_t cuda_error = cudaDeviceSynchronize();
  if (cuda_error != cudaSuccess) {
    std::cerr << "  " << description << ": " << cudaGetErrorString(cuda_error) << "\n";
    return false;
  }

  std::vector<Element> host_d(reference.size());
  cuda_error = cudaMemcpy(
      host_d.data(), args.ptr_d, sizeof(Element) * host_d.size(), cudaMemcpyDeviceToHost);
  if (cuda_error != cudaSuccess) {
    std::cerr << "  " << description << ": D2H failed: " << cudaGetErrorString(cuda_error) << "\n";
    return false;
  }

  double mae = 0.0;
  double max_abs = 0.0;
  for (size_t i = 0; i < reference.size(); ++i) {
    double diff = std::abs(double(float(host_d[i])) - double(float(reference[i])));
    mae += diff;
    max_abs = std::max(max_abs, diff);
  }
  mae /= double(reference.size());

  bool passed = mae <= double(options.mae_tolerance) && std::isfinite(mae);

  std::cout << "  " << description << "\n"
            << "    MAE      : " << mae << " (tolerance " << options.mae_tolerance << ")\n"
            << "    max |err|: " << max_abs << "\n"
            << "    result   : " << (passed ? "PASS" : "FAIL") << "\n";

  if (!passed) {
    return false;
  }

  if (options.iterations > 0) {
    // Explicit warmup: bring the GPU up to steady-state clocks before timing.
    // Without this the first kernel measured absorbs the clock ramp and reads
    // several times slower than it actually is.
    for (int iter = 0; iter < kWarmupIterations; ++iter) {
      op.run(nullptr);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Only kernel launches inside the loop -- no re-initialization.
    cudaEventRecord(start);
    for (int iter = 0; iter < options.iterations; ++iter) {
      op.run(nullptr);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    std::cout << "    avg time : " << avg_runtime_ms << " ms\n"
              << "    GFLOP/s  : " << options.gflops(avg_runtime_ms / 1000.0) << "\n";
  }

  return true;
}

int run(Options const& options) {
  int const M = options.m;
  int const N = options.n;
  int const K = options.k;

  std::cout << "EVT-fused residual-block GEMM  D = ((A.B + bias) + C1) + C2\n"
            << "    problem  : " << M << " x " << N << " x " << K << "\n"
            << "    element  : half_t, accumulate/compute in float\n"
            << "    tile     : 128x128x32, warp 64x64x32, mma 16x8x16, 4 stages\n"
            << "    EVT tree : AuxStore(D) <- plus(C2) <- plus(C1) <- plus(AccFetch, RowBroadcast(bias))\n";

  std::vector<Element> host_a(size_t(M) * K);
  std::vector<Element> host_b(size_t(K) * N);
  // Braces, not parens: `std::vector<Element> host_bias(size_t(N))` is a
  // most-vexing-parse function declaration, not a vector.
  std::vector<Element> host_bias(static_cast<size_t>(N), Element(0));
  std::vector<Element> host_c1(size_t(M) * N);
  std::vector<Element> host_c2(size_t(M) * N);

  // Small deterministic values: FP16 accumulation of a K-long dot product needs
  // the inputs kept small for the FP32 host reference to stay a fair comparison.
  testing::fill_deterministic(host_a, 0.02f, 0.10f);
  testing::fill_deterministic(host_b, 0.02f, 0.35f);
  testing::fill_deterministic(host_bias, 0.05f, 0.70f);
  testing::fill_deterministic(host_c1, 0.05f, 1.10f);
  testing::fill_deterministic(host_c2, 0.05f, 1.70f);

  testing::DeviceBuffer<Element> d_a(host_a.size());
  testing::DeviceBuffer<Element> d_b(host_b.size());
  testing::DeviceBuffer<Element> d_bias(host_bias.size());
  testing::DeviceBuffer<Element> d_c1(host_c1.size());
  testing::DeviceBuffer<Element> d_c2(host_c2.size());
  testing::DeviceBuffer<Element> d_d(size_t(M) * N);

  if (!d_a.copy_from_host(host_a) || !d_b.copy_from_host(host_b) ||
      !d_bias.copy_from_host(host_bias) || !d_c1.copy_from_host(host_c1) ||
      !d_c2.copy_from_host(host_c2)) {
    return 1;
  }

  std::cout << "    reference: unfused device GEMM + separate elementwise kernel\n";
  std::vector<Element> reference;
  if (!reference_device(
          options, d_a.get(), d_b.get(), d_bias.get(), d_c1.get(), d_c2.get(), reference)) {
    return 1;
  }

  auto make_args = [&](int avail_sms) {
    typename DataParallelOp::Arguments args;
    args.problem_size = cutlass::gemm::GemmCoord(M, N, K);
    args.ptr_a = d_a.get();
    args.ptr_b = d_b.get();
    args.ptr_bias = d_bias.get();
    args.ptr_c1 = d_c1.get();
    args.ptr_c2 = d_c2.get();
    args.ptr_d = d_d.get();
    args.split_k_slices = options.split_k_slices;
    args.avail_sms = avail_sms;
    return args;
  };

  bool all_passed = true;
  double dp_ms = 0.0;
  double sk_ms = 0.0;

  // Data-parallel: -1 means "not a StreamK kernel, ignore avail_sms".
  cudaMemset(d_d.get(), 0, sizeof(Element) * size_t(M) * N);
  all_passed &= run_one<DataParallelOp>(
      "data-parallel (GemmIdentityThreadblockSwizzle)", options, make_args(-1), reference, dp_ms);

  // StreamK: same tree, same arguments layout, different swizzle.
  cudaMemset(d_d.get(), 0, sizeof(Element) * size_t(M) * N);
  typename StreamKOp::Arguments sk_args;
  {
    auto base = make_args(-1);
    sk_args.problem_size = base.problem_size;
    sk_args.ptr_a = base.ptr_a;
    sk_args.ptr_b = base.ptr_b;
    sk_args.ptr_bias = base.ptr_bias;
    sk_args.ptr_c1 = base.ptr_c1;
    sk_args.ptr_c2 = base.ptr_c2;
    sk_args.ptr_d = base.ptr_d;
    sk_args.split_k_slices = 1;
    sk_args.avail_sms = -1;
  }
  all_passed &= run_one<StreamKOp>(
      "StreamK (ThreadblockSwizzleStreamK)", options, sk_args, reference, sk_ms);

  if (options.iterations > 0 && dp_ms > 0.0 && sk_ms > 0.0) {
    std::cout << "\n    StreamK / data-parallel time ratio: " << (sk_ms / dp_ms) << "\n";
  }

  std::cout << "\n" << (all_passed ? "PASS" : "FAIL") << "\n";
  return all_passed ? 0 : 1;
}

} // namespace
} // namespace evt
} // namespace tiny_cutlass

int main(int argc, char const** argv) {
  tiny_cutlass::evt::Options options;
  options.parse(argc, argv);

  if (options.help) {
    tiny_cutlass::evt::Options::print_usage(std::cout);
    return 0;
  }
  if (options.error) {
    std::cerr << "Invalid arguments.\n";
    tiny_cutlass::evt::Options::print_usage(std::cerr);
    return 1;
  }

  cudaDeviceProp props;
  int device_id = 0;
  if (cudaGetDevice(&device_id) != cudaSuccess ||
      cudaGetDeviceProperties(&props, device_id) != cudaSuccess) {
    std::cerr << "Failed to query the CUDA device.\n";
    return 1;
  }
  if (props.major * 10 + props.minor < 80) {
    std::cerr << "This kernel requires SM80 or newer (found SM" << props.major << props.minor
              << ").\n";
    return 1;
  }

  return tiny_cutlass::evt::run(options);
}
