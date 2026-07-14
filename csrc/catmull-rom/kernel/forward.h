#pragma once

#include <cuda_runtime.h>

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/half.h"

namespace tiny_cutlass::catmull_rom::kernel {

// ---------------------------------------------------------------------------
// Catmull-Rom reprojection kernel -- faithful translation of Unreal
// Engine's Bicubic2DCatmullRom (Engine/Shaders/PostProcessTemporalAA.usf).
//
// TAA / temporal super-resolution reprojects each target pixel back into the
// previous ("history") frame through its motion vector, landing at a
// fractional location pi(p) = p + mv(p). A Catmull-Rom cubic spline is the
// standard interpolating filter used there because bilinear resampling softens
// high-frequency content and the blur accumulates across frames.
//
// UE uses the Keys cubic with A = -0.5, expressed as (f = fractional offset):
//   w0 = f^2 - 0.5 (f^3 + f)
//   w1 = 1.5 f^3 - 2.5 f^2 + 1
//   w3 = 0.5 (f^3 - f^2)
//   w2 = 1 - w0 - w1 - w3
// The CUDA gather evaluates the full 16-tap separable form.
//
// Layout: history / output are NHWC [N, H, W, C]. motion is [N, H, W, 2] with
// per-pixel (dx, dy) in pixel units; sampled source location is (x+dx, y+dy).
// Out-of-bounds taps use clamp-to-edge addressing.
// ---------------------------------------------------------------------------

template <typename Element_, typename ElementMotion_>
struct CatmullRomReprojectParams {
  using Element = Element_;
  using ElementMotion = ElementMotion_;

  Element const* history = nullptr;       // [N, H, W, C]
  ElementMotion const* motion = nullptr;  // [N, H, W, 2] (dx, dy) in pixels
  Element* output = nullptr;              // [N, H, W, C]

  int N = 0;
  int H = 0;
  int W = 0;
  int C = 0;
};

// UE Bicubic2DCatmullRom weights (Keys cubic, A = -0.5) for the four taps at
// integer offsets {-1, 0, +1, +2} around floor(sample), given fractional part
// f in [0, 1). Sum to 1; interpolating (reproduces samples at tap centers).
CUTLASS_HOST_DEVICE
void catmull_rom_weights(float f, float w[4]) {
  float f2 = f * f;
  float f3 = f2 * f;
  w[0] = f2 - 0.5f * (f3 + f);
  w[1] = 1.5f * f3 - 2.5f * f2 + 1.0f;
  w[3] = 0.5f * (f3 - f2);
  w[2] = 1.0f - w[0] - w[1] - w[3];
}

CUTLASS_HOST_DEVICE
int clamp_index(int v, int hi) {
  if (v < 0) return 0;
  if (v > hi) return hi;
  return v;
}

// Core per-pixel-per-channel reprojection math, shared by the forward image
// kernel and by the Slang cross-check test kernel. Given the 4x4 clamped tap
// values (row-major, taps[j*4 + i] = column i of row j) and the fractional
// offsets (tx, ty), returns the separable Catmull-Rom interpolation.
CUTLASS_HOST_DEVICE
float catmull_rom_interp16(float const taps[16], float tx, float ty) {
  float wx[4];
  float wy[4];
  catmull_rom_weights(tx, wx);
  catmull_rom_weights(ty, wy);

  float acc = 0.0f;
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < 4; ++j) {
    float row = 0.0f;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      row += wx[i] * taps[j * 4 + i];
    }
    acc += wy[j] * row;
  }
  return acc;
}

// One block per output pixel; block threads stride over the channel axis so
// per-channel gathers are coalesced across NHWC-contiguous channels.
template <typename Params>
__global__ void catmull_rom_reproject_kernel(Params params) {
  using Element = typename Params::Element;

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

  float wx[4];
  float wy[4];
  catmull_rom_weights(tx, wx);
  catmull_rom_weights(ty, wy);

  int xs[4];
  int ys[4];
  CUTLASS_PRAGMA_UNROLL
  for (int k = 0; k < 4; ++k) {
    xs[k] = clamp_index(ix - 1 + k, W - 1);
    ys[k] = clamp_index(iy - 1 + k, H - 1);
  }

  int64_t const plane = static_cast<int64_t>(n) * H;
  int64_t const out_base = ((plane + y) * W + x) * static_cast<int64_t>(C);

  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    float acc = 0.0f;
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < 4; ++j) {
      int64_t const row_base = ((plane + ys[j]) * W) * static_cast<int64_t>(C);
      float row_acc = 0.0f;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < 4; ++i) {
        int64_t const idx = row_base + static_cast<int64_t>(xs[i]) * C + c;
        row_acc += wx[i] * static_cast<float>(params.history[idx]);
      }
      acc += wy[j] * row_acc;
    }
    params.output[out_base + c] = Element(acc);
  }
}

// Kernel-layer policy factory: bundles element types and launch shape for one
// architecture, and exposes the Params type plus the __global__ entry.
template <
    typename ArchTag = cutlass::arch::Sm80,
    typename Element = cutlass::half_t,
    typename ElementMotion = float,
    int ThreadsPerBlock = 128>
struct DefaultCatmullRomReproject {
  using Params = CatmullRomReprojectParams<Element, ElementMotion>;

  static int const kThreadsPerBlock = ThreadsPerBlock;

  static void invoke(Params const& params, cudaStream_t stream) {
    int const total_pixels = params.N * params.H * params.W;
    if (total_pixels <= 0) {
      return;
    }
    dim3 const grid(static_cast<unsigned>(total_pixels), 1, 1);
    dim3 const block(static_cast<unsigned>(kThreadsPerBlock), 1, 1);
    catmull_rom_reproject_kernel<Params><<<grid, block, 0, stream>>>(params);
  }
};

}  // namespace tiny_cutlass::catmull_rom::kernel
