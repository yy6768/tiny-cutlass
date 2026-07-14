#pragma once

#include <cuda_runtime.h>

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/half.h"

#include "catmull-rom/kernel/forward.h"

namespace tiny_cutlass::catmull_rom::kernel {

// ---------------------------------------------------------------------------
// Backward pass for the UE Catmull-Rom reprojection (see
// forward.h for the forward and the UE weight formulas).
//
// Forward:
//   out[p,c] = sum_j sum_i wy[j] * wx[i] * history[ys[j], xs[i], c]
// with wx = CR(tx), wy = CR(ty), tx/ty the fractional parts of (x+mvx, y+mvy),
// and xs/ys clamped to edge.
//
// Given g = grad_output this kernel produces (a single kernel, both grads):
//   grad_history[ys[j], xs[i], c] += wy[j] * wx[i] * g[p,c]           (scatter)
//   grad_motion.x[p] = sum_c g[p,c] * sum_j sum_i wy[j] * dwx[i] * history (mv.x)
//   grad_motion.y[p] = sum_c g[p,c] * sum_j sum_i dwy[j] * wx[i] * history (mv.y)
// where dwx/dwy are the CR weight derivatives w.r.t. the fractional offset
// (dtx/dmvx = 1).
//
// grad_history aliases across taps whenever clamping collapses neighboring
// taps onto the same edge pixel, so the scatter uses atomicAdd. Grad buffers
// are fp32 (ElementAccum). grad_history MUST be zeroed before launch (the
// device op does this in run()).
// ---------------------------------------------------------------------------

// Derivative of the UE Catmull-Rom weights (A = -0.5) w.r.t. f. Since the four
// weights sum to 1 for all f, the four derivatives sum to 0.
CUTLASS_HOST_DEVICE
void catmull_rom_weight_derivs(float f, float dw[4]) {
  float f2 = f * f;
  dw[0] = -1.5f * f2 + 2.0f * f - 0.5f;
  dw[1] = 4.5f * f2 - 5.0f * f;
  dw[3] = 1.5f * f2 - f;
  dw[2] = -(dw[0] + dw[1] + dw[3]);  // = -4.5 f^2 + 4 f + 0.5
}

template <typename Element_, typename ElementMotion_, typename ElementAccum_ = float>
struct CatmullRomReprojectBackwardParams {
  using Element = Element_;
  using ElementMotion = ElementMotion_;
  using ElementAccum = ElementAccum_;

  Element const* history = nullptr;        // [N, H, W, C]
  ElementMotion const* motion = nullptr;   // [N, H, W, 2]
  Element const* grad_output = nullptr;    // [N, H, W, C]

  ElementAccum* grad_history = nullptr;    // [N, H, W, C], pre-zeroed
  ElementAccum* grad_motion = nullptr;     // [N, H, W, 2]

  int N = 0;
  int H = 0;
  int W = 0;
  int C = 0;
};

// One block per output pixel; threads stride over channels. Each thread does
// the grad_history scatter for its channels and accumulates that channel's
// contribution to (gmx, gmy); a shared-memory tree reduction (ThreadsPerBlock
// must be a power of two) produces the single per-pixel motion gradient.
template <typename Params, int ThreadsPerBlock>
__global__ void catmull_rom_reproject_backward_kernel(Params params) {
  using Element = typename Params::Element;
  using ElementAccum = typename Params::ElementAccum;

  int const pixel = blockIdx.x;
  int const total_pixels = params.N * params.H * params.W;
  if (pixel >= total_pixels) {
    return;
  }

  int const W = params.W;
  int const H = params.H;
  int const C = params.C;

  int const x = pixel % W;
  int const row = pixel / W;
  int const y = row % H;
  int const n = row / H;

  float const mvx = static_cast<float>(params.motion[pixel * 2 + 0]);
  float const mvy = static_cast<float>(params.motion[pixel * 2 + 1]);

  float const fx = static_cast<float>(x) + mvx;
  float const fy = static_cast<float>(y) + mvy;
  int const ix = static_cast<int>(floorf(fx));
  int const iy = static_cast<int>(floorf(fy));
  float const tx = fx - static_cast<float>(ix);
  float const ty = fy - static_cast<float>(iy);

  float wx[4], wy[4], dwx[4], dwy[4];
  catmull_rom_weights(tx, wx);
  catmull_rom_weights(ty, wy);
  catmull_rom_weight_derivs(tx, dwx);
  catmull_rom_weight_derivs(ty, dwy);

  int xs[4], ys[4];
  CUTLASS_PRAGMA_UNROLL
  for (int k = 0; k < 4; ++k) {
    xs[k] = clamp_index(ix - 1 + k, W - 1);
    ys[k] = clamp_index(iy - 1 + k, H - 1);
  }

  int64_t const plane = static_cast<int64_t>(n) * H;
  int64_t const out_base = ((plane + y) * W + x) * static_cast<int64_t>(C);

  float gmx = 0.0f;
  float gmy = 0.0f;

  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    float const g = static_cast<float>(params.grad_output[out_base + c]);
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < 4; ++j) {
      int64_t const row_base = ((plane + ys[j]) * W) * static_cast<int64_t>(C);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < 4; ++i) {
        int64_t const idx = row_base + static_cast<int64_t>(xs[i]) * C + c;
        float const h = static_cast<float>(params.history[idx]);
        atomicAdd(&params.grad_history[idx], static_cast<ElementAccum>(wy[j] * wx[i] * g));
        gmx += g * wy[j] * dwx[i] * h;
        gmy += g * dwy[j] * wx[i] * h;
      }
    }
  }

  __shared__ float s_gmx[ThreadsPerBlock];
  __shared__ float s_gmy[ThreadsPerBlock];
  s_gmx[threadIdx.x] = gmx;
  s_gmy[threadIdx.x] = gmy;
  __syncthreads();

  CUTLASS_PRAGMA_NO_UNROLL
  for (int stride = ThreadsPerBlock / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_gmx[threadIdx.x] += s_gmx[threadIdx.x + stride];
      s_gmy[threadIdx.x] += s_gmy[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    params.grad_motion[pixel * 2 + 0] = static_cast<ElementAccum>(s_gmx[0]);
    params.grad_motion[pixel * 2 + 1] = static_cast<ElementAccum>(s_gmy[0]);
  }
}

template <
    typename ArchTag = cutlass::arch::Sm80,
    typename Element = cutlass::half_t,
    typename ElementMotion = float,
    typename ElementAccum = float,
    int ThreadsPerBlock = 128>
struct DefaultCatmullRomReprojectBackward {
  using Params = CatmullRomReprojectBackwardParams<Element, ElementMotion, ElementAccum>;

  static int const kThreadsPerBlock = ThreadsPerBlock;

  static void invoke(Params const& params, cudaStream_t stream) {
    int const total_pixels = params.N * params.H * params.W;
    if (total_pixels <= 0) {
      return;
    }
    dim3 const grid(static_cast<unsigned>(total_pixels), 1, 1);
    dim3 const block(static_cast<unsigned>(kThreadsPerBlock), 1, 1);
    catmull_rom_reproject_backward_kernel<Params, ThreadsPerBlock>
        <<<grid, block, 0, stream>>>(params);
  }
};

}  // namespace tiny_cutlass::catmull_rom::kernel
