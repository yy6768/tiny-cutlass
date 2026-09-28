#pragma once

#include <cuda_fp16.h>
#include "cutlass/aligned_buffer.h"
#include "swin/encoder/ops/encoder.h"
#include "swin/encoder/thread/activation.h"
#include "swin/encoder/threadblock/gemm.h"

namespace tiny_cutlass::swin::encoder::threadblock {

// Custom fusion and reductions recovered from the supplied PTX. Matrix
// products are delegated to CUTLASS warp MMA and its standard iterators.
template <typename WarpMma_>
struct Encoder {
  using WarpMma = WarpMma_;
  using Element = typename WarpMma::ElementA;
  using MatrixMultiply = Gemm<WarpMma>;
  using Params = Arguments<Element>;
  using Exponential = thread::EncodedExponential<Element>;
  using Activation = thread::ClippedGelu<Element>;

  struct SharedStorage {
    typename MatrixMultiply::SharedStorage mma;
    cutlass::AlignedBuffer<Element, 96 * 32> features, normalized, q, v, temporary;
    cutlass::AlignedBuffer<Element, 64 * 96> probability;
    cutlass::AlignedBuffer<Element, 64 * 32> residual;
  };

  CUTLASS_DEVICE
  void operator()(Params const& args, SharedStorage& s, int tile) const {
    int lane = int(threadIdx.x);
    auto const& w = args.weights;
    MatrixMultiply mma;
    // Initialization belongs to the CTA mainloop. No separate Loader layer.
    for (int i = lane; i < 96 * 32; i += 32) {
      s.features.data()[i] = args.input[int64_t(tile) * 96 * 32 + i];
      s.temporary.data()[i] = w.input_bias[0][i % 32];
    }
    __syncthreads();
    mma(s.mma, s.features.data(), 32, w.input_weight[0], 32, false,
        s.temporary.data(), 32, 96, 32, 32);
    for (int i = lane; i < 96 * 32; i += 32) {
      s.temporary.data()[i] = Element(__hmax(s.temporary.data()[i].to_half(), __float2half(0)));
      s.features.data()[i] = w.input_bias[1][i % 32];
    }
    __syncthreads();
    mma(s.mma, s.temporary.data(), 32, w.input_weight[1], 32, false,
        s.features.data(), 32, 96, 32, 32);

    // Half sum-of-squares norm (NOT mean-square RMSNorm). The reduction tree
    // follows the PTX fragment grouping: [c,c+16], [c+8,c+24], xor 1/2, halves.
    for (int stage = 0; stage < 2; ++stage) {
      Element* source = stage == 0 ? s.features.data() : s.residual.data();
      int rows = stage == 0 ? 96 : 64;
      for (int row = lane; row < rows; row += 32) {
        __half sums[4][2];
        for (int group = 0; group < 4; ++group) {
          for (int part = 0; part < 2; ++part) {
            int c = 2 * group + part;
            __half a = source[row * 32 + c].to_half();
            __half b = source[row * 32 + c + 8].to_half();
            __half d = source[row * 32 + c + 16].to_half();
            __half e = source[row * 32 + c + 24].to_half();
            sums[group][part] = __hadd(__hadd(__hmul(a, a), __hmul(d, d)),
                                      __hadd(__hmul(b, b), __hmul(e, e)));
          }
        }
        __half sum[2];
        for (int part = 0; part < 2; ++part)
          sum[part] = __hadd(__hadd(sums[0][part], sums[1][part]),
                            __hadd(sums[2][part], sums[3][part]));
        __half inverse = hrsqrt(__hadd(__hadd(sum[0], sum[1]), __float2half(0.0001220703125f)));
        for (int c = 0; c < 32; ++c) {
          __half gain = __hmul(inverse, w.norm_weight[stage][c].to_half());
          s.normalized.data()[row * 32 + c] = Element(__hmul(source[row * 32 + c].to_half(), gain));
        }
      }
      __syncthreads();
      if (stage == 1) break;

      for (int i = lane; i < 64 * 32; i += 32)
        s.residual.data()[i] = Element(__hadd(s.features.data()[i].to_half(),
                                             w.projection_bias[i % 32].to_half()));
      __syncthreads();
      for (int head = 0; head < 2; ++head) {
        for (int i = lane; i < 96 * 32; i += 32) {
          s.q.data()[i] = Element(0);
          s.v.data()[i] = Element(0);
        }
        for (int i = lane; i < 64 * 96; i += 32)
          s.probability.data()[i] = w.position_bias[head][i];
        __syncthreads();
        mma(s.mma, s.normalized.data(), 32, w.qk_weight[head], 32, false,
            s.q.data(), 32, 96, 32, 32);
        mma(s.mma, s.normalized.data(), 32, w.value_weight[head], 32, false,
            s.v.data(), 32, 96, 32, 32);
        mma(s.mma, s.q.data(), 32, s.q.data(), 32, false,
            s.probability.data(), 96, 64, 96, 32);
        for (int row = lane; row < 64; row += 32) {
          __half values[96];
          for (int col = 0; col < 96; ++col)
            values[col] = Exponential{}(s.probability.data()[row * 96 + col]).to_half();
          __half partial[4][2];
          for (int group = 0; group < 4; ++group) {
            for (int part = 0; part < 2; ++part) {
              int c = group * 2 + part;
              __half a = __hadd(__hadd(values[c], values[c + 16]),
                                __hadd(values[c + 32], values[c + 48]));
              a = __hadd(a, __hadd(values[c + 64], values[c + 80]));
              __half b = __hadd(__hadd(values[c + 8], values[c + 24]),
                                __hadd(values[c + 40], values[c + 56]));
              b = __hadd(b, __hadd(values[c + 72], values[c + 88]));
              partial[group][part] = __hadd(a, b);
            }
          }
          __half sums[2];
          for (int part = 0; part < 2; ++part)
            sums[part] = __hadd(__hadd(partial[0][part], partial[1][part]),
                               __hadd(partial[2][part], partial[3][part]));
          __half inverse = hrcp(__hadd(sums[0], sums[1]));
          for (int col = 0; col < 96; ++col)
            s.probability.data()[row * 96 + col] = Element(__hmul(values[col], inverse));
        }
        for (int i = lane; i < 64 * 32; i += 32) s.temporary.data()[i] = Element(0);
        __syncthreads();
        mma(s.mma, s.probability.data(), 96, s.v.data(), 32, true,
            s.temporary.data(), 32, 64, 32, 96);
        mma(s.mma, s.temporary.data(), 32, w.projection_weight[head], 32, false,
            s.residual.data(), 32, 64, 32, 32);
      }
    }

    for (int i = lane; i < 64 * 32; i += 32)
      s.residual.data()[i] = Element(__hadd(s.residual.data()[i].to_half(), w.contract_bias[i % 32].to_half()));
    __syncthreads();
    for (int chunk = 0; chunk < 4; ++chunk) {
      for (int i = lane; i < 64 * 32; i += 32) s.temporary.data()[i] = w.expand_bias[chunk][i % 32];
      __syncthreads();
      mma(s.mma, s.normalized.data(), 32, w.expand_weight[chunk], 32, false,
          s.temporary.data(), 32, 64, 32, 32);
      for (int i = lane; i < 64 * 32; i += 32)
        s.temporary.data()[i] = Activation{}(s.temporary.data()[i]);
      __syncthreads();
      mma(s.mma, s.temporary.data(), 32, w.contract_weight[chunk], 32, false,
          s.residual.data(), 32, 64, 32, 32);
    }
    for (int i = lane; i < 64 * 32; i += 32)
      args.output[int64_t(tile) * 64 * 32 + i] = s.residual.data()[i];

    // Canonical 2x2 concatenation: TL, TR, BL, BR. The weight importer defines
    // the corresponding K permutation; there is no additional merging norm.
    for (int i = lane; i < 16 * 128; i += 32) {
      int row = i / 128, k = i % 128, pixel = k / 32;
      int y = (row / 4) * 2 + pixel / 2, x = (row % 4) * 2 + pixel % 2;
      s.q.data()[i] = s.residual.data()[(y * 8 + x) * 32 + k % 32];
    }
    for (int i = lane; i < 16 * 64; i += 32) s.temporary.data()[i] = w.merge_bias[i % 64];
    __syncthreads();
    mma(s.mma, s.q.data(), 128, w.merge_weight, 128, false,
        s.temporary.data(), 64, 16, 64, 128);
    for (int i = lane; i < 16 * 64; i += 32)
      args.merged[int64_t(tile) * 16 * 64 + i] = s.temporary.data()[i];

    // Pad the 16-output projection to the existing 32-column warp tile.
    for (int i = lane; i < 32 * 32; i += 32)
      s.q.data()[i] = i < 16 * 32 ? w.head_weight[i] : Element(0);
    for (int i = lane; i < 64 * 32; i += 32)
      s.temporary.data()[i] = i % 32 < 16 ? w.head_bias[i % 32] : Element(0);
    __syncthreads();
    mma(s.mma, s.residual.data(), 32, s.q.data(), 32, false,
        s.temporary.data(), 32, 64, 32, 32);
    for (int i = lane; i < 64 * 16; i += 32)
      args.head[int64_t(tile) * 64 * 16 + i] = s.temporary.data()[(i / 16) * 32 + i % 16];
  }
};

}  // namespace tiny_cutlass::swin::encoder::threadblock
