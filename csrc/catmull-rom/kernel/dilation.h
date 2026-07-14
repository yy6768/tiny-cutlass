#pragma once

#include <cuda_runtime.h>

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/half.h"

#include "catmull-rom/kernel/forward.h"

namespace tiny_cutlass::catmull_rom::kernel {

// ---------------------------------------------------------------------------
// Depth-based motion dilation.
//
// A standard TAA pre-pass: before reprojecting the history buffer, each pixel
// replaces its own motion vector with the motion vector of the *nearest*
// neighbor in a small cross-shaped neighborhood. This "dilates" the motion of
// foreground (nearer) objects outward across their silhouette, so that pixels
// just outside a moving object still reproject with the object's motion rather
// than the (stale) background motion -- removing the ghost/tear artifacts at
// moving-object edges.
//
// Neighborhood: 5-tap cross -- center (0,0) plus up/down/left/right (+/-1).
// For each pixel we pick the tap with the largest "nearness" and copy that
// tap's (dx, dy) into the output. Depth semantics are configurable via
// NearerIsGreater: when true a *larger* depth value means nearer (the caller's
// convention here); when false a *smaller* depth means nearer (classic z/w
// depth buffers). Edge taps are clamped to the center pixel.
//
// Layout: depth is [N, H, W] (single channel). motion / output are
// [N, H, W, 2] with (dx, dy) in pixel units.
// ---------------------------------------------------------------------------

template <typename ElementDepth_, typename ElementMotion_, bool NearerIsGreater_ = true>
struct MotionDilationParams {
  using ElementDepth = ElementDepth_;
  using ElementMotion = ElementMotion_;
  static bool const kNearerIsGreater = NearerIsGreater_;

  ElementDepth const* depth = nullptr;    // [N, H, W]
  ElementMotion const* motion = nullptr;  // [N, H, W, 2]
  ElementMotion* output = nullptr;        // [N, H, W, 2] dilated motion

  int N = 0;
  int H = 0;
  int W = 0;
};

// Returns true if candidate depth `cand` is "nearer" than the current best
// `best` under the configured convention.
template <bool NearerIsGreater>
CUTLASS_HOST_DEVICE bool depth_is_nearer(float cand, float best) {
  return NearerIsGreater ? (cand > best) : (cand < best);
}

// One thread per pixel. Reads the 5-tap cross of depth, picks the nearest tap,
// and copies that tap's motion vector. Cheap enough that a flat 1D grid with a
// standard block size is fine; no shared memory needed for the standalone op.
template <typename Params>
__global__ void motion_dilation_kernel(Params params) {
  using ElementMotion = typename Params::ElementMotion;

  int const total_pixels = params.N * params.H * params.W;
  int const pixel = blockIdx.x * blockDim.x + threadIdx.x;
  if (pixel >= total_pixels) {
    return;
  }

  int const W = params.W;
  int const H = params.H;

  int const x = pixel % W;
  int const row = pixel / W;
  int const y = row % H;
  int const n = row / H;

  int64_t const plane = static_cast<int64_t>(n) * H;

  // 5-tap cross offsets: center, left, right, up, down.
  int const dx_off[5] = {0, -1, 1, 0, 0};
  int const dy_off[5] = {0, 0, 0, -1, 1};

  int best_pixel = pixel;
  float best_depth = static_cast<float>(params.depth[pixel]);

  CUTLASS_PRAGMA_UNROLL
  for (int k = 1; k < 5; ++k) {
    int const nx = x + dx_off[k];
    int const ny = y + dy_off[k];
    if (nx < 0 || nx >= W || ny < 0 || ny >= H) {
      continue;  // clamp-to-center: skip out-of-bounds taps.
    }
    int const npix = static_cast<int>((plane + ny) * W + nx);
    float const d = static_cast<float>(params.depth[npix]);
    if (depth_is_nearer<Params::kNearerIsGreater>(d, best_depth)) {
      best_depth = d;
      best_pixel = npix;
    }
  }

  params.output[pixel * 2 + 0] = params.motion[best_pixel * 2 + 0];
  params.output[pixel * 2 + 1] = params.motion[best_pixel * 2 + 1];
}

template <
    typename ArchTag = cutlass::arch::Sm80,
    typename ElementDepth = float,
    typename ElementMotion = float,
    bool NearerIsGreater = true,
    int ThreadsPerBlock = 256>
struct DefaultMotionDilation {
  using Params = MotionDilationParams<ElementDepth, ElementMotion, NearerIsGreater>;

  static int const kThreadsPerBlock = ThreadsPerBlock;

  static void invoke(Params const& params, cudaStream_t stream) {
    int const total_pixels = params.N * params.H * params.W;
    if (total_pixels <= 0) {
      return;
    }
    int const blocks = (total_pixels + kThreadsPerBlock - 1) / kThreadsPerBlock;
    dim3 const grid(static_cast<unsigned>(blocks), 1, 1);
    dim3 const block(static_cast<unsigned>(kThreadsPerBlock), 1, 1);
    motion_dilation_kernel<Params><<<grid, block, 0, stream>>>(params);
  }
};

// ---------------------------------------------------------------------------
// Fused depth-based motion dilation + Catmull-Rom reprojection.
//
// The full TAA reprojection pipeline in a single kernel, with no intermediate
// dilated-motion buffer written to global memory:
//   1. Depth dilation: for the current pixel, sample the 5-tap depth cross
//      (center + up/down/left/right) and pick the nearest tap's motion vector
//      (see dilation.h for the depth-convention discussion).
//   2. Catmull-Rom gather: use that dilated motion to backward-reproject and
//      resample the history buffer with the 16-tap separable Keys cubic
//      (see forward.h).
//
// One block per output pixel. The dilation is a per-pixel decision, so thread 0
// computes it once and publishes the chosen (dx, dy) through shared memory;
// every channel thread then reads the same motion and runs the gather. This
// avoids recomputing the depth cross per channel and avoids a global-memory
// round-trip for the dilated motion.
//
// Layout: depth is [N, H, W]; history / output are NHWC [N, H, W, C]; motion
// is [N, H, W, 2] with (dx, dy) in pixel units.
// ---------------------------------------------------------------------------

template <
    typename Element_,
    typename ElementDepth_,
    typename ElementMotion_,
    bool NearerIsGreater_ = true>
struct DilatedCatmullReprojectParams {
  using Element = Element_;
  using ElementDepth = ElementDepth_;
  using ElementMotion = ElementMotion_;
  static bool const kNearerIsGreater = NearerIsGreater_;

  Element const* history = nullptr;       // [N, H, W, C]
  ElementDepth const* depth = nullptr;    // [N, H, W]
  ElementMotion const* motion = nullptr;  // [N, H, W, 2]
  Element* output = nullptr;              // [N, H, W, C]

  int N = 0;
  int H = 0;
  int W = 0;
  int C = 0;
};

template <typename Params>
__global__ void dilated_catmull_reproject_kernel(Params params) {
  using Element = typename Params::Element;
  using ElementMotion = typename Params::ElementMotion;

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

  int64_t const plane = static_cast<int64_t>(n) * H;

  // --- Step 1: depth dilation (thread 0 only), published via shared memory ---
  __shared__ float s_mvx;
  __shared__ float s_mvy;

  if (threadIdx.x == 0) {
    int const dx_off[5] = {0, -1, 1, 0, 0};
    int const dy_off[5] = {0, 0, 0, -1, 1};

    int best_pixel = pixel;
    float best_depth = static_cast<float>(params.depth[pixel]);

    CUTLASS_PRAGMA_UNROLL
    for (int k = 1; k < 5; ++k) {
      int const nx = x + dx_off[k];
      int const ny = y + dy_off[k];
      if (nx < 0 || nx >= W || ny < 0 || ny >= H) {
        continue;
      }
      int const npix = static_cast<int>((plane + ny) * W + nx);
      float const d = static_cast<float>(params.depth[npix]);
      if (depth_is_nearer<Params::kNearerIsGreater>(d, best_depth)) {
        best_depth = d;
        best_pixel = npix;
      }
    }

    s_mvx = static_cast<float>(params.motion[best_pixel * 2 + 0]);
    s_mvy = static_cast<float>(params.motion[best_pixel * 2 + 1]);
  }
  __syncthreads();

  // --- Step 2: Catmull-Rom gather with the dilated motion ---
  float const fx = static_cast<float>(x) + s_mvx;
  float const fy = static_cast<float>(y) + s_mvy;
  int const ix = static_cast<int>(floorf(fx));
  int const iy = static_cast<int>(floorf(fy));

  float wx[4];
  float wy[4];
  catmull_rom_weights(fx - static_cast<float>(ix), wx);
  catmull_rom_weights(fy - static_cast<float>(iy), wy);

  int xs[4];
  int ys[4];
  CUTLASS_PRAGMA_UNROLL
  for (int k = 0; k < 4; ++k) {
    xs[k] = clamp_index(ix - 1 + k, W - 1);
    ys[k] = clamp_index(iy - 1 + k, H - 1);
  }

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

template <
    typename ArchTag = cutlass::arch::Sm80,
    typename Element = cutlass::half_t,
    typename ElementDepth = float,
    typename ElementMotion = float,
    bool NearerIsGreater = true,
    int ThreadsPerBlock = 128>
struct DefaultDilatedCatmullReproject {
  using Params =
      DilatedCatmullReprojectParams<Element, ElementDepth, ElementMotion, NearerIsGreater>;

  static int const kThreadsPerBlock = ThreadsPerBlock;

  static void invoke(Params const& params, cudaStream_t stream) {
    int const total_pixels = params.N * params.H * params.W;
    if (total_pixels <= 0) {
      return;
    }
    dim3 const grid(static_cast<unsigned>(total_pixels), 1, 1);
    dim3 const block(static_cast<unsigned>(kThreadsPerBlock), 1, 1);
    dilated_catmull_reproject_kernel<Params><<<grid, block, 0, stream>>>(params);
  }
};

}  // namespace tiny_cutlass::catmull_rom::kernel

