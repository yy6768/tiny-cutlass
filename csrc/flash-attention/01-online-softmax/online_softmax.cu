/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

/*! \file
    \brief Online Softmax Kernel.

    One row per block. Each thread processes a chunk of columns.
    Online algorithm (Milakov & Gimelshein, 2018): a single pass computes
    max and sum simultaneously by maintaining the recurrence

        m_new = max(m, x)
        d_new = d * exp(m - m_new) + exp(x - m_new)

    Cross-thread merge of two partial states (m1, d1), (m2, d2):

        m = max(m1, m2)
        d = d1 * exp(m1 - m) + d2 * exp(m2 - m)

    Final result needs one more pass to normalize: out[c] = exp(x - m) / d.
*/

#include <cfloat>

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Merge two (max, sum) partial states into the lhs.
__device__ __forceinline__
void online_merge(float& m, float& d, float m_other, float d_other) {
  float m_new = fmaxf(m, m_other);
  // Guard against m == m_other == -inf (empty state) to avoid -inf + -inf -> NaN.
  if (m_new == -FLT_MAX) {
    m = m_new;
    d = 0.0f;
    return;
  }
  d = d * __expf(m - m_new) + d_other * __expf(m_other - m_new);
  m = m_new;
}

/// Warp-level reduction of (m, d) using butterfly shuffles.
__device__ __forceinline__
void online_warp_reduce(float& m, float& d) {
  #pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    float m_other = __shfl_xor_sync(0xffffffff, m, offset);
    float d_other = __shfl_xor_sync(0xffffffff, d, offset);
    online_merge(m, d, m_other, d_other);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Online softmax kernel — one row per threadblock.
///
/// shared memory layout: 2 * num_warps floats
///   [0 .. num_warps)              : per-warp max
///   [num_warps .. 2 * num_warps)  : per-warp sum
__global__ void online_softmax_kernel(
    float const* __restrict__ input,    // [rows, cols], row-major
    float*       __restrict__ output,   // [rows, cols], row-major
    int rows,
    int cols
) {
  int row = blockIdx.x;
  if (row >= rows) return;

  float const* row_in  = input  + row * cols;
  float*       row_out = output + row * cols;

  int num_warps = (blockDim.x + warpSize - 1) / warpSize;
  int warp_id   = threadIdx.x / warpSize;
  int lane_id   = threadIdx.x % warpSize;

  extern __shared__ float smem[];
  float* smem_max = smem;                  // [num_warps]
  float* smem_sum = smem + num_warps;      // [num_warps]

  // -------------------------------------------------------------------------
  // Pass 1: per-thread online recurrence over assigned columns.
  // -------------------------------------------------------------------------
  float m = -FLT_MAX;
  float d = 0.0f;

  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float x = row_in[c];
    online_merge(m, d, x, 1.0f);  // exp(x - x) == 1
  }

  // Warp-level merge.
  online_warp_reduce(m, d);

  // Lane 0 of each warp publishes its (m, d) to shared memory.
  if (lane_id == 0) {
    smem_max[warp_id] = m;
    smem_sum[warp_id] = d;
  }
  __syncthreads();

  // Warp 0 reduces across warps.
  if (warp_id == 0) {
    if (lane_id < num_warps) {
      m = smem_max[lane_id];
      d = smem_sum[lane_id];
    } else {
      m = -FLT_MAX;
      d = 0.0f;
    }
    online_warp_reduce(m, d);

    if (lane_id == 0) {
      smem_max[0] = m;
      smem_sum[0] = d;
    }
  }
  __syncthreads();

  float row_max = smem_max[0];
  float inv_sum = 1.0f / smem_sum[0];

  // -------------------------------------------------------------------------
  // Pass 2: normalize.
  // -------------------------------------------------------------------------
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    row_out[c] = __expf(row_in[c] - row_max) * inv_sum;
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
