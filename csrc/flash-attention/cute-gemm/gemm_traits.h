/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

/*! \file
    \brief Compile-time configuration for the CuTe warm-up GEMM.

    Problem contract (matches csrc/tests/flash-attention/cute_gemm_test.cpp):

      C = alpha * A * B^T + beta * C

      A: (M, K) row-major   -- K contiguous
      B: (N, K) row-major   -- K contiguous
      C: (M, N) row-major   -- N contiguous

    This is the CuTe tutorial's "TN" case (3rdparty/cutlass/examples/cute/
    tutorial/sgemm_sm80.cu, gemm_tn): both operands are K-major in gmem, which
    is what lets a single swizzled smem atom serve both A and B.

    Every tile shape, layout, and atom below is a fixed decision -- nothing
    here is a TODO. gemm_kernel.h is the file where the mainloop itself is
    left for you to write; this file only hands it the types to write it
    with.
*/

#pragma once

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

namespace cute_gemm {

using namespace cute;

// FP16 in, FP32 accumulate -- matches the cuBLAS reference
// (CUBLAS_COMPUTE_32F) in the shared test harness.
using GemmElement = cutlass::half_t;
using GemmAccumulator = float;

struct GemmProblem {
  int m = 0;
  int n = 0;
  int k = 0;
  float alpha = 1.0f;
  float beta = 0.0f;
};

struct GemmTensors {
  GemmElement const* a = nullptr; // (M, K) row-major
  GemmElement const* b = nullptr; // (N, K) row-major
  GemmElement* c = nullptr;       // (M, N) row-major
};

// ---- CTA tile shape --------------------------------------------------------
// 128x128x64, triple-buffered over K. Same shape as the tutorial's SM80
// TN kernel; large enough to be bandwidth-friendly, small enough that one
// CTA's smem footprint comfortably fits Ampere's 164KB/SM budget at bP=3.
using TileM = Int<128>;
using TileN = Int<128>;
using TileK = Int<64>;
using PipelineStages = Int<3>;

using CtaTiler = decltype(make_shape(TileM{}, TileN{}, TileK{}));

// ---- Shared memory layouts --------------------------------------------------
// Swizzle<3,3,3> composed with an 8x(8x8) atom removes bank conflicts on the
// 128-bit ldmatrix loads below; tile_to_shape repeats it out to
// (TileM|TileN, TileK, PipelineStages).
using SmemSwizzleAtom = decltype(composition(
    Swizzle<3, 3, 3>{},
    Layout<Shape<_8, Shape<_8, _8>>, Stride<_8, Stride<_1, _64>>>{}));

using SmemLayoutA = decltype(tile_to_shape(
    SmemSwizzleAtom{}, make_shape(TileM{}, TileK{}, PipelineStages{})));
using SmemLayoutB = decltype(tile_to_shape(
    SmemSwizzleAtom{}, make_shape(TileN{}, TileK{}, PipelineStages{})));
using SmemLayoutC = decltype(make_layout(make_shape(TileM{}, TileN{})));

// ---- Gmem <-> smem copy (cp.async, 128-bit vectorized) ----------------------
using GmemTiledCopyA = decltype(make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, GemmElement>{},
    Layout<Shape<_16, _8>, Stride<_8, _1>>{}, // 16x8 thread layout, K-major
    Layout<Shape<_1, _8>>{}));                // 1x8 values per thread, K-major
using GmemTiledCopyB = GmemTiledCopyA;

// ---- Tensor-core MMA atom + tiling ------------------------------------------
// SM80 m16n8k16, FP32 accumulate. 2x2 atoms over a 32x32x16 tile -> 4 warps.
using MmaAtom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
using TiledMma = decltype(make_tiled_mma(
    MmaAtom{}, Layout<Shape<_2, _2>>{}, Tile<_32, _32, _16>{}));

// ---- Smem -> register copy for the MMA operands (ldmatrix) ------------------
using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, GemmElement>;
using SmemCopyAtomB = Copy_Atom<SM75_U32x4_LDSM_N, GemmElement>;

static constexpr int kThreadCount = 128; // size(TiledMma{})

} // namespace cute_gemm
