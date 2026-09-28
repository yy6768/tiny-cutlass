#pragma once
#include "cutlass/aligned_buffer.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h"
#include "swin/fused_swin_layer/threadblock/gemm.h"
#include <math_constants.h>

namespace tiny_cutlass::swin::fused_swin_layer::threadblock {

template <typename Mma0, typename Mma1, typename Mma2, typename Epilogue_>
struct FusedSwinLayer {
  using Element = typename Mma0::IteratorA::Element;
  using Epilogue = Epilogue_;
  using Gemm0 = Gemm<Mma0, Epilogue, false>;
  using Gemm1 = Gemm<Mma1, Epilogue, true>;
  using Gemm2 = Gemm<Mma2, Epilogue, true>;
  using ThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<
      cutlass::layout::PitchLinearShape<32, 1>, 32, 1>;
  template <typename T>
  using RowIterator = cutlass::transform::threadblock::RegularTileIterator<
      cutlass::MatrixShape<1, 32>, T, cutlass::layout::RowMajor, 1, ThreadMap>;
  template <typename T>
  using GlobalRowIterator = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<1, 32>, T, cutlass::layout::RowMajor, 1, ThreadMap, 1>;
  using InputIterator = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<16, 32>, Element, cutlass::layout::RowMajor, 1,
      typename Mma0::IteratorA::ThreadMap, 8, true>;

  struct SharedStorage {
    union {
      typename Gemm0::SharedStorage gemm0;
      typename Gemm1::SharedStorage gemm1;
      typename Gemm2::SharedStorage gemm2;
    } gemm;
    cutlass::AlignedBuffer<Element, 16 * 32> input, residual, normalized, probability;
    // Padding belongs to ordinary logical tensors; MMA storage/layout comes from MmaBase.
    cutlass::AlignedBuffer<Element, 32 * 32> q, k, v;
    cutlass::AlignedBuffer<Element, 16 * 128> hidden;
    cutlass::AlignedBuffer<float, 16 * 32> accum;
  };

  template <typename Arguments>
  CUTLASS_DEVICE
  void operator()(Arguments const& args, SharedStorage& shared, int window) const {
    int lane = int(threadIdx.x);
    using Row = RowIterator<Element>;
    using FloatRow = RowIterator<float>;
    using GlobalRow = GlobalRowIterator<Element>;
    typename GlobalRow::Params row_params{cutlass::layout::RowMajor(32)};
    typename Row::Fragment x, gamma, r, value, bias;
    typename FloatRow::Fragment result;

    typename InputIterator::Params input_params{cutlass::layout::RowMajor(32)};
    InputIterator input(input_params, const_cast<Element*>(args.input), {16, 32},
                        lane, {0, 0}, args.gather + int64_t(window) * 16);
    typename InputIterator::Fragment input_fragment;
    input.load(input_fragment);
    typename Mma0::IteratorA shared_input({shared.input.data(), cutlass::layout::RowMajor(32)}, lane);
    shared_input.store(input_fragment);
    GlobalRow gamma_iterator(row_params, const_cast<Element*>(args.rms_weight), {1, 32}, lane);
    gamma_iterator.load(gamma);
    __syncthreads();

    // RMSNorm1 over all 32 channels. One lane per channel, all tensor reads use iterators.
    for (int row = 0; row < 16; ++row) {
      Row source({shared.input.data() + row * 32, cutlass::layout::RowMajor(32)}, lane);
      source.load(x);
      float sum = float(x[0]) * float(x[0]);
      CUTLASS_PRAGMA_UNROLL
      for (int delta = 16; delta; delta /= 2) sum += __shfl_xor_sync(0xffffffff, sum, delta);
      value[0] = Element(float(x[0]) * rsqrtf(sum / 32 + args.problem.rms_epsilon) * float(gamma[0]));
      Row destination({shared.normalized.data() + row * 32, cutlass::layout::RowMajor(32)}, lane);
      destination.store(value);
    }
    value.clear();
    for (int row = 16; row < 32; ++row) {
      Row q({shared.q.data() + row * 32, cutlass::layout::RowMajor(32)}, lane);
      Row k({shared.k.data() + row * 32, cutlass::layout::RowMajor(32)}, lane);
      Row v({shared.v.data() + row * 32, cutlass::layout::RowMajor(32)}, lane);
      q.store(value); k.store(value); v.store(value);
    }
    __syncthreads();

    // Packed bias-free Q/V or Q/K/V, weights are [output_channel, input_channel].
    int projections = args.problem.separate_qk ? 3 : 2;
    for (int projection = 0; projection < projections; ++projection) {
      Gemm0{}(shared.gemm.gemm0, shared.normalized.data(), 32,
              args.qkv_weight + projection * 32 * 32, 32, 32, shared.accum.data());
      Element* output = projection == 0 ? shared.q.data() :
          (projection == projections - 1 ? shared.v.data() : shared.k.data());
      for (int row = 0; row < 16; ++row) {
        FloatRow source({shared.accum.data() + row * 32, cutlass::layout::RowMajor(32)}, lane);
        source.load(result); value[0] = Element(result[0]);
        Row destination({output + row * 32, cutlass::layout::RowMajor(32)}, lane);
        destination.store(value);
      }
      __syncthreads();
    }

    Gemm1{}(shared.gemm.gemm1, shared.q.data(), 32,
            args.problem.separate_qk ? shared.k.data() : shared.q.data(),
            32, 32, shared.accum.data());
    for (int row = 0; row < 16; ++row) {
      FloatRow source({shared.accum.data() + row * 32, cutlass::layout::RowMajor(32)}, lane);
      source.load(result);
      GlobalRow position(row_params, const_cast<Element*>(args.position_bias + row * 16), {1, 16}, lane);
      bias.clear(); position.load(bias);
      float score = float(Element(float(Element(result[0])) + float(bias[0])));
      score = lane < 16 ? float(Element(score * rsqrtf(32.0f))) : -CUDART_INF_F;
      float maximum = score;
      CUTLASS_PRAGMA_UNROLL
      for (int delta = 16; delta; delta /= 2)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, delta));
      float probability = expf(score - maximum), sum = probability;
      CUTLASS_PRAGMA_UNROLL
      for (int delta = 16; delta; delta /= 2) sum += __shfl_xor_sync(0xffffffff, sum, delta);
      value[0] = Element(probability / sum);
      Row destination({shared.probability.data() + row * 32, cutlass::layout::RowMajor(32)}, lane);
      destination.store(value);
    }
    __syncthreads();
    Gemm2{}(shared.gemm.gemm2, shared.probability.data(), 32, shared.v.data(), 32, 32, shared.accum.data());
    for (int row = 0; row < 16; ++row) {
      FloatRow source({shared.accum.data() + row * 32, cutlass::layout::RowMajor(32)}, lane);
      source.load(result); value[0] = Element(result[0]);
      Row destination({shared.normalized.data() + row * 32, cutlass::layout::RowMajor(32)}, lane);
      destination.store(value);
    }
    __syncthreads();
    Gemm0{}(shared.gemm.gemm0, shared.normalized.data(), 32, args.output_weight, 32, 32, shared.accum.data());
    bias.clear();
    if (args.output_bias) {
      GlobalRow output_bias(row_params, const_cast<Element*>(args.output_bias), {1, 32}, lane);
      output_bias.load(bias);
    }
    GlobalRow mlp_gamma(row_params, const_cast<Element*>(args.mlp_rms_weight), {1, 32}, lane);
    mlp_gamma.load(gamma);
    for (int row = 0; row < 16; ++row) {
      FloatRow source({shared.accum.data() + row * 32, cutlass::layout::RowMajor(32)}, lane);
      source.load(result);
      Row input_row({shared.input.data() + row * 32, cutlass::layout::RowMajor(32)}, lane);
      input_row.load(x);
      r[0] = Element(float(Element(result[0] + float(bias[0]))) + float(x[0]));
      Row residual({shared.residual.data() + row * 32, cutlass::layout::RowMajor(32)}, lane);
      residual.store(r);
      float sum = float(r[0]) * float(r[0]);
      CUTLASS_PRAGMA_UNROLL
      for (int delta = 16; delta; delta /= 2) sum += __shfl_xor_sync(0xffffffff, sum, delta);
      value[0] = Element(float(r[0]) * rsqrtf(sum / 32 + args.problem.mlp_rms_epsilon) * float(gamma[0]));
      Row destination({shared.normalized.data() + row * 32, cutlass::layout::RowMajor(32)}, lane);
      destination.store(value);
    }
    __syncthreads();
    for (int column = 0; column < 128; column += 32) {
      Gemm0{}(shared.gemm.gemm0, shared.normalized.data(), 32,
              args.fc1_weight + column * 32, 32, 32, shared.accum.data());
      GlobalRow fc1_bias(row_params, const_cast<Element*>(args.fc1_bias + column), {1, 32}, lane);
      fc1_bias.load(bias);
      for (int row = 0; row < 16; ++row) {
        FloatRow source({shared.accum.data() + row * 32, cutlass::layout::RowMajor(32)}, lane);
        source.load(result);
        float z = float(Element(result[0] + float(bias[0])));
        value[0] = Element(cutlass::epilogue::thread::GELU<float>{}(z));
        Row destination({shared.hidden.data() + row * 128 + column, cutlass::layout::RowMajor(128)}, lane);
        destination.store(value);
      }
      __syncthreads();
    }
    Gemm0{}(shared.gemm.gemm0, shared.hidden.data(), 128, args.fc2_weight, 128, 128, shared.accum.data());
    GlobalRow fc2_bias(row_params, const_cast<Element*>(args.fc2_bias), {1, 32}, lane);
    fc2_bias.load(bias);
    using IndexIterator = GlobalRowIterator<int>;
    typename IndexIterator::Params index_params{cutlass::layout::RowMajor(16)};
    IndexIterator scatter(index_params, const_cast<int*>(args.scatter + int64_t(window) * 16), {1, 16}, lane);
    typename IndexIterator::Fragment index; index.clear(); scatter.load(index);
    for (int row = 0; row < 16; ++row) {
      int output_row = __shfl_sync(0xffffffff, index[0], row);
      if (output_row >= 0) {
        FloatRow source({shared.accum.data() + row * 32, cutlass::layout::RowMajor(32)}, lane);
        source.load(result);
        Row residual({shared.residual.data() + row * 32, cutlass::layout::RowMajor(32)}, lane);
        residual.load(r);
        value[0] = Element(float(Element(result[0] + float(bias[0]))) + float(r[0]));
        GlobalRow destination(row_params, args.output + int64_t(output_row) * 32, {1, 32}, lane);
        destination.store(value);
      }
    }
  }
};

}  // namespace tiny_cutlass::swin::fused_swin_layer::threadblock
