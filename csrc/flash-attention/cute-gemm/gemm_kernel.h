/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

/*! \file
    \brief CuTe warm-up GEMM device kernel -- mainloop body is a learning
    exercise. Fill in the TODOs below; nothing outside this file needs to
    change to make it correct.

    Everything you need is already a compile-time type from gemm_traits.h:
    tile shape, smem layouts (already swizzled), gmem<->smem tiled copies,
    the TiledMma, and the smem->register ldmatrix atoms. Your job is only the
    control flow that stitches them together.

    Read 3rdparty/cutlass/examples/cute/tutorial/sgemm_sm80.cu (gemm_device)
    for the reference shape of this loop -- same tile config, same operand
    layout (TN), same cp.async pipelining idea. Don't copy-paste it; work out
    each step from the CuTe primitives so it actually sinks in.
*/

#pragma once

#include "cute/tensor.hpp"

#include "gemm_traits.h"

namespace cute_gemm {

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA, class S2RAtomA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB, class S2RAtomB,
          class TC, class CStride, class CSmemLayout, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a, S2RAtomA s2r_atom_a,
            TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b, S2RAtomB s2r_atom_b,
            TC      * C, CStride dC, CSmemLayout          , TiledMma mma,
            Alpha alpha, Beta beta) {
  using namespace cute;

  // ------------------------------------------------------------------------
  // Step 1: build the full-problem gmem tensors, then take this CTA's tile.
  //
  //   Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
  //   Tensor mB = ...                                                       // (N,K)
  //   Tensor mC = ...                                                       // (M,N)
  //
  // Then local_tile(..., cta_tiler, cta_coord, Step<...>{}) to get gA/gB/gC
  // for this block. cta_coord comes from blockIdx: make_coord(blockIdx.x,
  // blockIdx.y, _) walks the K mode with `_` (keep all k-tiles).
  //
  // Think about which Step<> mask each of gA/gB/gC needs -- gA varies over
  // (M,K), gB varies over (N,K), gC varies over (M,N). X marks the mode each
  // tensor does NOT have.
  // ------------------------------------------------------------------------

  // TODO Step 1: mA/mB/mC, cta_coord, gA/gB/gC.

  // ------------------------------------------------------------------------
  // Step 2: allocate the two smem buffers (extern __shared__ + reinterpret,
  // or plain __shared__ arrays sized by cosize_v<ASmemLayout>/<BSmemLayout>
  // -- either is fine here since A/B don't need to overlap with anything
  // else). Wrap each in make_tensor(make_smem_ptr(...), sA_layout /
  // sB_layout) to get sA, sB.
  // ------------------------------------------------------------------------

  // TODO Step 2: sA, sB.

  // ------------------------------------------------------------------------
  // Step 3: partition the gmem->smem copy. copy_a/copy_b are TiledCopy
  // objects; get_slice(threadIdx.x) gives you this thread's ThrCopy, and
  // partition_S/partition_D split gA/gB (source) and sA/sB (dest) into this
  // thread's piece: tAgA, tAsA, tBgB, tBsB.
  // ------------------------------------------------------------------------

  // TODO Step 3: thr_copy_a/thr_copy_b, tAgA/tAsA, tBgB/tBsB.

  // ------------------------------------------------------------------------
  // Step 4: prime the smem pipeline. PipelineStages (bP=3) means you can
  // have up to 2 outstanding async loads in flight before you need the
  // first tile's data. Issue cp.async for k_pipe = 0 .. K_PIPE_MAX-2,
  // fencing after each pair, and track which gmem k-tile you're on
  // (k_tile_next) and how many remain (k_tile_count = size<3>(tAgA)).
  // ------------------------------------------------------------------------

  // TODO Step 4: K_PIPE_MAX, k_tile_count, k_tile_next, initial cp_async loop.

  // ------------------------------------------------------------------------
  // Step 5: set up the MMA-side partitioning and accumulator.
  //   ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  //   Tensor tCgC = thr_mma.partition_C(gC);
  //   Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));
  //   Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));
  //   Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  //   clear(tCrC);
  // ------------------------------------------------------------------------

  // TODO Step 5: thr_mma, tCgC, tCrA/tCrB, tCrC, clear.

  // ------------------------------------------------------------------------
  // Step 6: retile the ldmatrix (smem->register) copy atoms against the MMA
  // partitioning, so tXrA/tXrB alias the same registers as tCrA/tCrB:
  //   TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
  //   ThrCopy s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
  //   Tensor tXsA = s2r_thr_copy_a.partition_S(sA);
  //   Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);
  // Same for B.
  // ------------------------------------------------------------------------

  // TODO Step 6: s2r_copy_a/s2r_copy_b, tXsA/tXrA, tXsB/tXrB.

  // ------------------------------------------------------------------------
  // Step 7: the pipelined main loop. Outer loop walks gmem k-tiles; inner
  // loop (k_block, size = size<2>(tCrA)) walks the smem tile in MMA-sized
  // K-chunks, register-pipelining one step ahead:
  //   - before the loop: if K_BLOCK_MAX > 1, cp_async_wait + __syncthreads,
  //     then copy the k_block=0 slice smem->regs to prime it.
  //   - each k_block: copy the *next* k_block's smem->regs slice, and on the
  //     last k_block of the tile issue the next tile's gmem->smem cp.async
  //     (advancing k_tile_next/k_tile_count and the smem pipe read/write
  //     indices), then gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC).
  //   - loop condition: k_tile_count > -(K_PIPE_MAX-1), so the last
  //     (K_PIPE_MAX-1) already-issued tiles still get consumed.
  //
  // This is the part worth working through by hand -- it's the same
  // register/smem double-buffering idea FA2 uses, just for a plain GEMM.
  // ------------------------------------------------------------------------

  // TODO Step 7: pipelined main loop.

  // ------------------------------------------------------------------------
  // Step 8: epilogue. axpby(alpha, tCrC, beta, tCgC) computes
  // tCgC = alpha * tCrC + beta * tCgC elementwise over the partitioned tile.
  // ------------------------------------------------------------------------

  // TODO Step 8: axpby(alpha, tCrC, beta, tCgC).
}

} // namespace cute_gemm
