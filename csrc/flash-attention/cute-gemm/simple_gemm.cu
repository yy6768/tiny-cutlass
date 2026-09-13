/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

/*! \file
    \brief Launch path for the CuTe warm-up GEMM: builds the dynamic strides
    and grid/block dims, then invokes gemm_kernel.h's gemm_device<<<>>>.

    Row-major contract (see gemm_traits.h): A is (M,K), B is (N,K), C is
    (M,N), all K/N contiguous. In CuTe's (stride0, stride1) convention that
    is dA=(K,1), dB=(K,1), dC=(N,1) -- the TN case, matching TiledMma's
    SM80_16x8x16_F32F16F16F32_TN atom.
*/

#include "simple_gemm.h"

#include "cute/tensor.hpp"

#include "gemm_kernel.h"
#include "gemm_traits.h"

namespace cute_gemm {

bool cute_simple_gemm_can_run(GemmProblem const& problem, std::string& reason) {
  if (problem.m <= 0 || problem.n <= 0 || problem.k <= 0) {
    reason = "m, n, k must all be positive";
    return false;
  }
  if (problem.k % int(TileK{}) != 0) {
    reason = "k must be a multiple of the K tile (64) -- this warm-up kernel does not predicate the K edge";
    return false;
  }
  return true;
}

char const* cute_simple_gemm_name() {
  return "CuTe warm-up GEMM (SM80, FP16xFP16->FP32, TN)";
}

cudaError_t cute_simple_gemm(
    GemmProblem const& problem,
    GemmTensors const& tensors,
    cudaStream_t stream) {
  using namespace cute;

  auto M = problem.m;
  auto N = problem.n;
  auto K = problem.k;
  auto prob_shape = make_shape(M, N, K);

  // TN strides: A is (M,K) row-major -> dA=(K,1); B is (N,K) row-major ->
  // dB=(K,1); C is (M,N) row-major -> dC=(N,1).
  auto dA = make_stride(K, Int<1>{});
  auto dB = make_stride(K, Int<1>{});
  auto dC = make_stride(N, Int<1>{});

  TiledMma mma{};
  GmemTiledCopyA copy_a{};
  GmemTiledCopyB copy_b{};
  SmemCopyAtomA s2r_atom_a{};
  SmemCopyAtomB s2r_atom_b{};

  dim3 grid_dim(size(ceil_div(M, TileM{})), size(ceil_div(N, TileN{})));
  dim3 block_dim(kThreadCount);

  auto* kernel = &gemm_device<
      decltype(prob_shape), CtaTiler,
      GemmElement, decltype(dA), SmemLayoutA, GmemTiledCopyA, SmemCopyAtomA,
      GemmElement, decltype(dB), SmemLayoutB, GmemTiledCopyB, SmemCopyAtomB,
      GemmElement, decltype(dC), SmemLayoutC, TiledMma,
      float, float>;

  gemm_device<<<grid_dim, block_dim, 0, stream>>>(
      prob_shape, CtaTiler{},
      tensors.a, dA, SmemLayoutA{}, copy_a, s2r_atom_a,
      tensors.b, dB, SmemLayoutB{}, copy_b, s2r_atom_b,
      tensors.c, dC, SmemLayoutC{}, mma,
      problem.alpha, problem.beta);

  (void)kernel;
  return cudaGetLastError();
}

} // namespace cute_gemm

bool cute_simple_gemm_can_run(GemmProblem const& problem, std::string& reason) {
  return cute_gemm::cute_simple_gemm_can_run(problem, reason);
}

char const* cute_simple_gemm_name() {
  return cute_gemm::cute_simple_gemm_name();
}

cudaError_t cute_simple_gemm(
    GemmProblem const& problem,
    GemmTensors const& tensors,
    cudaStream_t stream) {
  return cute_gemm::cute_simple_gemm(problem, tensors, stream);
}
