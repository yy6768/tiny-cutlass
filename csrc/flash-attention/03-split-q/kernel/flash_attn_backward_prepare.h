#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"
#include "../../flash_attention.h"

// FA2's custom Delta prepass is a row dot product, not a replacement for any
// of the five TensorOp matrix products. O is the public rounded forward output.
template <class Element_, int ElementsPerAccess_, int ThreadsPerRow_>
struct FlashAttnBackwardDelta {
  using Element = Element_;
  using AccessType = cutlass::Array<Element, ElementsPerAccess_>;
  static constexpr int kElementsPerAccess = ElementsPerAccess_;
  static constexpr int kThreadsPerRow = ThreadsPerRow_;
  static constexpr int kThreadCount = 128;
  static constexpr int kRowsPerBlock = kThreadCount / kThreadsPerRow;
  static_assert(sizeof(AccessType) == 16, "Delta uses 128-bit vector loads");
  static_assert(kThreadsPerRow >= 2 && kThreadsPerRow <= 32 &&
      (kThreadsPerRow & (kThreadsPerRow - 1)) == 0,
      "Each Delta row must occupy a power-of-two warp subgroup");
  struct SharedStorage {};
  struct Params {
    Problem problem;
    Element const* output;
    Element const* grad_output;
    float* delta;
    int64_t rows;
  };

  CUTLASS_DEVICE
  void operator()(Params const& params, SharedStorage&) {
    auto const& p = params.problem;
    int worker = int(threadIdx.x) % kThreadsPerRow;
    // FA2 flash_bwd_preprocess_kernel.h:36-46 accumulates each worker's
    // vectors before the descending XOR tree in utils.h:111-130.
    // The row-block loop is CTA-uniform so tail rows also execute every shuffle.
    for (int64_t row_base = int64_t(blockIdx.x) * kRowsPerBlock;
         row_base < params.rows; row_base += int64_t(gridDim.x) * kRowsPerBlock) {
      int64_t row = row_base + int(threadIdx.x) / kThreadsPerRow;
      bool valid_row = row < params.rows;
      int64_t offset = 0;
      if (valid_row) {
        int q = int(row % p.seq_length);
        int64_t bh = row / p.seq_length;
        int head = int(bh % p.head_number);
        int64_t batch = bh / p.head_number;
        offset = ((batch * p.seq_length + q) * p.head_number + head) * p.head_size;
      }
      float sum = 0.f;
      for (int page = 0; page < p.head_size;
           page += kThreadsPerRow * kElementsPerAccess) {
        int column = page + worker * kElementsPerAccess;
        bool valid = valid_row && column + kElementsPerAccess <= p.head_size;
        int64_t vector_offset = valid ? offset + column : 0;
        AccessType output, grad_output;
        output.clear();
        grad_output.clear();
        cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
            output, params.output + vector_offset, valid);
        cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
            grad_output, params.grad_output + vector_offset, valid);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kElementsPerAccess; ++i)
          sum += float(output[i]) * float(grad_output[i]);
      }
      CUTLASS_PRAGMA_UNROLL
      for (int distance = kThreadsPerRow / 2; distance > 0; distance /= 2)
        sum += __shfl_xor_sync(0xffffffff, sum, distance);
      if (valid_row && worker == 0) params.delta[row] = sum;
    }
  }
};

template <class Element_>
struct FlashAttnBackwardConvert {
  using Element = Element_;
  using Converter = cutlass::NumericConverter<Element, float>;
  static constexpr int kThreadCount = 256;
  struct SharedStorage {};
  struct Params {
    float const* input;
    Element* output;
    int64_t elements;
  };

  CUTLASS_DEVICE
  void operator()(Params const& params, SharedStorage&) {
    Converter convert;
    for (int64_t i = int64_t(blockIdx.x) * kThreadCount + threadIdx.x;
         i < params.elements; i += int64_t(gridDim.x) * kThreadCount)
      params.output[i] = convert(params.input[i]);
  }
};
