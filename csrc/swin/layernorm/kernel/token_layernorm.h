#pragma once

/*
  Row-wise LayerNorm over the channel axis of a [rows, cols] token tensor.

  ---------------------------------------------------------------------------
  WHY THIS IS A SEPARATE KERNEL (and patch_embed's LayerNorm is not)
  ---------------------------------------------------------------------------
  patch_embed's LayerNorm follows its GEMM, so it lives in the epilogue where a
  complete output row is already in registers (see
  swin/patch_embed/epilogue/layernorm_visitor.h).

  The LayerNorms in swin_mlp (LN2 before fc1) and patch_merging (norm before the
  4C -> 2C projection) come BEFORE their GEMM, and they normalize along the
  channel axis -- which for those GEMMs is the K (reduction) axis. A CTA walks K
  across many mainloop iterations, so at the moment operand A is first loaded no
  thread has seen the whole row, and the statistics do not exist yet. That is
  structural, not an artifact of this code: it is exactly why CUTLASS ships
  GemmLayernormMainloopFusion, which takes mean/variance PRECOMPUTED by a prior
  reduction pass rather than computing them inline.

  So a pre-GEMM LayerNorm is inherently its own pass. Both families that need
  one share this kernel.

  ---------------------------------------------------------------------------
  MAPPING
  ---------------------------------------------------------------------------
  One warp owns one row. Each lane strides across the row in vectors of
  kElementsPerAccess, accumulating sum and sum-of-squares in fp32, then a full
  32-lane butterfly completes both totals. Rows are independent, so a CTA holds
  kWarpsPerBlock warps working on kWarpsPerBlock consecutive rows.

  Numerics match the host reference and PyTorch:
    * fp32 accumulation regardless of storage type
    * BIASED variance (divide by N)
    * var = E[x^2] - E[x]^2, clamped at zero against cancellation
*/

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

namespace tiny_cutlass::swin::layernorm::kernel {

/// Parameters for the token-wise LayerNorm.
///
/// The optional SEGMENTED GATHER lets an output row be assembled from several
/// input rows concatenated along the channel axis:
///
///     out[r, k * segment_cols + c] = input[gather[k][r], c]
///
/// This is exactly PatchMerging's 2x2 concat (4 segments of C channels each).
/// Because the LayerNorm has to be its own pass anyway, folding the gather into
/// its read path makes the concat free -- no separate materialization of the
/// [out_tokens, 4C] tensor.
///
/// With num_segments == 1 and gather == nullptr this degenerates to a plain
/// row-wise LayerNorm, which is what swin_mlp's LN2 uses.
template <typename Element_, typename ElementCompute_>
struct TokenLayerNormParams {
  using Element = Element_;
  using ElementCompute = ElementCompute_;

  Element const* input = nullptr;
  Element* output = nullptr;
  ElementCompute const* gamma = nullptr;  // [cols], may be null => 1
  ElementCompute const* beta = nullptr;   // [cols], may be null => 0
  int rows = 0;
  int cols = 0;  // total normalized width == num_segments * segment_cols
  int64_t input_row_stride = 0;   // in elements
  int64_t output_row_stride = 0;  // in elements
  ElementCompute epsilon = ElementCompute(1e-5f);

  // Segmented gather. gather is [num_segments, rows]; segment k supplies
  // channels [k * segment_cols, (k+1) * segment_cols).
  int const* gather = nullptr;
  int num_segments = 1;
  int segment_cols = 0;  // defaults to cols when there is one segment
};

/// One warp per row; kWarpsPerBlock rows per CTA.
template <
    typename Element,
    typename ElementCompute,
    int ElementsPerAccess = 8,
    int WarpsPerBlock = 4>
__global__ void token_layernorm_kernel(
    TokenLayerNormParams<Element, ElementCompute> params) {
  static int const kElementsPerAccess = ElementsPerAccess;
  static int const kWarpsPerBlock = WarpsPerBlock;
  static int const kLanes = 32;

  using AccessType = cutlass::Array<Element, kElementsPerAccess>;
  using ComputeFragment = cutlass::Array<ElementCompute, kElementsPerAccess>;

  int const warp_id = int(threadIdx.x) / kLanes;
  int const lane_id = int(threadIdx.x) % kLanes;
  int const row = int(blockIdx.x) * kWarpsPerBlock + warp_id;

  if (row >= params.rows) {
    return;
  }

  Element* row_out = params.output + int64_t(row) * params.output_row_stride;

  int const vectors = params.cols / kElementsPerAccess;
  int const segment_cols =
      params.segment_cols > 0 ? params.segment_cols : params.cols;
  int const vectors_per_segment = segment_cols / kElementsPerAccess;

  // Resolves vector index v (over the whole normalized width) to its address,
  // following the segmented gather when one is configured.
  auto source_vector = [&](int v) -> AccessType const* {
    if (params.gather == nullptr) {
      Element const* base =
          params.input + int64_t(row) * params.input_row_stride;
      return reinterpret_cast<AccessType const*>(base) + v;
    }
    int const segment = v / vectors_per_segment;
    int const within = v - segment * vectors_per_segment;
    int const src_row = params.gather[segment * params.rows + row];
    Element const* base =
        params.input + int64_t(src_row) * params.input_row_stride;
    return reinterpret_cast<AccessType const*>(base) + within;
  };

  cutlass::NumericArrayConverter<ElementCompute, Element, kElementsPerAccess>
      to_compute;
  cutlass::NumericArrayConverter<Element, ElementCompute, kElementsPerAccess>
      to_element;

  // Pass 1: sum and sum-of-squares in fp32.
  ElementCompute sum = ElementCompute(0);
  ElementCompute sum_sq = ElementCompute(0);

  for (int v = lane_id; v < vectors; v += kLanes) {
    ComputeFragment value = to_compute(*source_vector(v));
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      sum += value[i];
      sum_sq += value[i] * value[i];
    }
  }

  // Butterfly across the full warp: every lane ends with the complete totals.
  CUTLASS_PRAGMA_UNROLL
  for (int offset = kLanes / 2; offset > 0; offset >>= 1) {
    sum += __shfl_xor_sync(0xFFFFFFFFu, sum, offset);
    sum_sq += __shfl_xor_sync(0xFFFFFFFFu, sum_sq, offset);
  }

  ElementCompute const inv_n = ElementCompute(1) / ElementCompute(params.cols);
  ElementCompute const mean = sum * inv_n;
  ElementCompute variance = sum_sq * inv_n - mean * mean;
  if (variance < ElementCompute(0)) {
    variance = ElementCompute(0);
  }
  ElementCompute const inv_std =
      ElementCompute(1) / cutlass::fast_sqrt(variance + params.epsilon);

  // Pass 2: normalize, scale, shift. Re-reads through the same gather, so the
  // concatenated tensor is never materialized.
  for (int v = lane_id; v < vectors; v += kLanes) {
    ComputeFragment value = to_compute(*source_vector(v));

    int const base = v * kElementsPerAccess;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      ElementCompute normalized = (value[i] - mean) * inv_std;
      if (params.gamma != nullptr) {
        normalized *= params.gamma[base + i];
      }
      if (params.beta != nullptr) {
        normalized += params.beta[base + i];
      }
      value[i] = normalized;
    }

    AccessType* dst = reinterpret_cast<AccessType*>(row_out) + v;
    *dst = to_element(value);
  }
}

}  // namespace tiny_cutlass::swin::layernorm::kernel
