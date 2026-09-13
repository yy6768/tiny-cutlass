#pragma once

#include "cutlass/aligned_buffer.h"
#include "swin/window_attention/ops/window_attention.h"
#include "swin/window_attention/warp/window_softmax.h"

namespace tiny_cutlass::swin::window_attention::threadblock {

// All four products use stock CUTLASS warp MMA and ldmatrix iterators.
// One warp processes a group at a time; groups may outnumber CTA warps.
template <typename WarpMma_, int Warps, int MaxQk, int MaxValue, int Access = 1>
struct WindowAttentionMma {
  using WarpMma = WarpMma_;
  using Element = typename WarpMma::ElementA;
  using LayoutA = typename WarpMma::LayoutA;
  using LayoutB = typename WarpMma::LayoutB;
  using Fragment = typename WarpMma::FragmentC;
  using Params = WindowAttentionArguments<Element>;
  static constexpr int kWarps = Warps;
  static constexpr int kMaxQk = MaxQk, kMaxValue = MaxValue;

  static constexpr int kAccess = Access;
  static_assert(Access == 1 || Access == 8, "unsupported operand access width");
  using AccessType = cutlass::AlignedArray<Element, Access, Access * sizeof(Element)>;
  CUTLASS_DEVICE
  static void copy(Element* dst, Element const* src, bool valid = true) {
    AccessType value;
    if (valid) value = *reinterpret_cast<AccessType const*>(src);
    else value.clear();
    *reinterpret_cast<AccessType*>(dst) = value;
  }
  struct WarpStorage {
    cutlass::AlignedBuffer<Element, 16 * 32> a;
    cutlass::AlignedBuffer<Element, 32 * 32> b;
    cutlass::AlignedBuffer<float, 16 * 32> accum;
    cutlass::AlignedBuffer<Element, 16 * MaxQk> q, k;
    cutlass::AlignedBuffer<Element, 16 * MaxValue> v;
    cutlass::AlignedBuffer<Element, 16 * 16> probability;
  };
  struct SharedStorage { WarpStorage warp[Warps]; };

  template <typename Storage>
  CUTLASS_DEVICE
  void mma_step(Storage& s, Fragment& accum, int lane) const {
    __syncwarp();
    typename WarpMma::IteratorA a({s.a.data(), LayoutA(32)}, lane);
    typename WarpMma::IteratorB b({s.b.data(), LayoutB(32)}, lane);
    typename WarpMma::FragmentA fa;
    typename WarpMma::FragmentB fb;
    WarpMma mma;
    #pragma unroll
    for (int k = 0; k < 32; k += 16) {
      a.load(fa); b.load(fb);
      ++a; ++b;
      mma(accum, fa, fb, accum);
    }
    __syncwarp();  // Finish iterator reads before overwriting shared operands.
  }

  template <typename Storage>
  CUTLASS_DEVICE
  void store(Storage& s, Fragment const& accum, int lane) const {
    typename WarpMma::IteratorC c({s.accum.data(), cutlass::layout::RowMajor(32)}, lane);
    c.store(accum);
    __syncwarp();
  }

  CUTLASS_DEVICE
  void operator()(Params const& args, SharedStorage& shared, int window,
                  Element* window_output = nullptr) const {
    auto const& p = args.problem;
    int lane = int(threadIdx.x) % 32, warp_id = int(threadIdx.x) / 32;
    auto& s = shared.warp[warp_id];
    Element* normalized = reinterpret_cast<Element*>(&shared + 1);
    Element* hidden = normalized + 16 * p.channels;
    LayoutA la(32); LayoutB lb(32);
    int const dq = p.qk_channels, dv = p.value_channels;
    int const out_channels = p.projected_channels(), hidden_channels = p.hidden_channels();
    int const* gather = args.gather + int64_t(window) * 16;
    int const* scatter = args.scatter + int64_t(window) * 16;

    // Normalize the complete channel row once, then reuse it for every group.
    // Reflect duplicates input rows but never changes their normalization axis.
    if constexpr (Access == 8) {
      int c = (lane % 4) * 8;
      AccessType gamma = *reinterpret_cast<AccessType const*>(args.rms_weight + c);
      for (int row = warp_id * 8 + lane / 4; row < 16; row += Warps * 8) {
        AccessType x = *reinterpret_cast<AccessType const*>(args.input + int64_t(gather[row]) * p.channels + c);
        float sum = 0;
        #pragma unroll
        for (int v = 0; v < 8; ++v) sum += float(x[v]) * float(x[v]);
        sum += __shfl_xor_sync(0xffffffff, sum, 1, 4);
        sum += __shfl_xor_sync(0xffffffff, sum, 2, 4);
        float inv = rsqrtf(sum / 32 + p.rms_epsilon);
        #pragma unroll
        for (int v = 0; v < 8; ++v) x[v] = Element(float(x[v]) * inv * float(gamma[v]));
        *reinterpret_cast<AccessType*>(normalized + row * p.channels + c) = x;
      }
    } else {
    for (int row = warp_id; row < 16; row += Warps) {
      float square_sum = 0;
      for (int c = lane; c < p.channels; c += 32) {
        float x = float(args.input[int64_t(gather[row]) * p.channels + c]);
        square_sum += x * x;
      }
      #pragma unroll
      for (int delta = 16; delta; delta /= 2)
        square_sum += __shfl_xor_sync(0xffffffff, square_sum, delta);
      float inv_rms = rsqrtf(square_sum / float(p.channels) + p.rms_epsilon);
      for (int c = lane; c < p.channels; c += 32) {
        float x = float(args.input[int64_t(gather[row]) * p.channels + c]);
        normalized[row * p.channels + c] = Element(x * inv_rms * float(args.rms_weight[c]));
      }
    }
    }
    __syncthreads();

    for (int group = warp_id; group < p.groups; group += Warps) {
      // Packed bias-free group projection. Partition rides in the input loads.
      for (int col = 0; col < out_channels; col += 32) {
        Fragment accum; accum.clear();
        for (int kc = 0; kc < p.channels; kc += 32) {
          for (int e = lane * Access; e < 16 * 32; e += 32 * Access) {
            int row = e / 32, k = e % 32;
            copy(s.a.data() + la({row, k}), normalized + row * p.channels + kc + k, kc + k < p.channels);
          }
          for (int e = lane * Access; e < 32 * 32; e += 32 * Access) {
            int n = e / 32, k = e % 32;
            copy(s.b.data() + lb({k, n}), args.qkv_weight + (int64_t(group) * out_channels + col + n) * p.channels + kc + k,
                 col + n < out_channels && kc + k < p.channels);
          }
          mma_step(s, accum, lane);
        }
        store(s, accum, lane);
        for (int e = lane * Access; e < 16 * 32; e += 32 * Access) {
          int row = e / 32, c = col + e % 32;
          AccessType value;
          #pragma unroll
          for (int v = 0; v < Access; ++v) value[v] = Element(s.accum.data()[e + v]);
          if (c < dq) *reinterpret_cast<AccessType*>(s.q.data() + row * MaxQk + c) = value;
          else if (p.qk_mode == QkMode::kSeparate && c < 2 * dq)
            *reinterpret_cast<AccessType*>(s.k.data() + row * MaxQk + c - dq) = value;
          else if (c < out_channels)
            *reinterpret_cast<AccessType*>(s.v.data() + row * MaxValue + c - (out_channels - dv)) = value;
        }
        __syncwarp();
      }

      Fragment scores; scores.clear();
      for (int kc = 0; kc < dq; kc += 32) {
        for (int e = lane * Access; e < 16 * 32; e += 32 * Access) {
          int row = e / 32, k = e % 32;
          copy(s.a.data() + la({row, k}), s.q.data() + row * MaxQk + kc + k, kc + k < dq);
        }
        for (int e = lane * Access; e < 32 * 32; e += 32 * Access) {
          int n = e / 32, k = e % 32;
          Element* key = p.qk_mode == QkMode::kShared ? s.q.data() : s.k.data();
          copy(s.b.data() + lb({k, n}), key + n * MaxQk + kc + k, n < 16 && kc + k < dq);
        }
        mma_step(s, scores, lane);
      }
      store(s, scores, lane);
      for (int e = lane; e < 256; e += 32) {
        int i = e / 16, j = e % 16;
        float bias = float(args.position_bias[group * 256 + e]);
        // The supplied eager-half reference rounds each materialized operation.
        float score = float(Element(float(Element(s.accum.data()[i * 32 + j])) + bias));
        s.accum.data()[i * 32 + j] = float(Element(score * rsqrtf(float(dq))));
      }
      __syncwarp();
      warp::WindowSoftmax<Element>{}(s.accum.data(), s.probability.data(), lane);
      __syncwarp();

      for (int col = 0; col < dv; col += 32) {
        Fragment accum; accum.clear();
        for (int e = lane * Access; e < 16 * 32; e += 32 * Access) {
          int row = e / 32, k = e % 32;
          copy(s.a.data() + la({row, k}), s.probability.data() + row * 16 + k, k < 16);
        }
        for (int e = lane * Access; e < 32 * 32; e += 32 * Access) {
          int n = e / 32, k = e % 32;
          AccessType value;
          #pragma unroll
          for (int v = 0; v < Access; ++v)
            value[v] = k + v < 16 && col + n < dv ? s.v.data()[(k + v) * MaxValue + col + n] : Element(0);
          *reinterpret_cast<AccessType*>(s.b.data() + lb({k, n})) = value;
        }
        mma_step(s, accum, lane);
        store(s, accum, lane);
        for (int e = lane * Access; e < 16 * 32; e += 32 * Access) {
          int row = e / 32, c = col + e % 32;
          if (c < dv) {
            AccessType value;
            #pragma unroll
            for (int v = 0; v < Access; ++v) value[v] = Element(s.accum.data()[e + v]);
            *reinterpret_cast<AccessType*>(hidden + row * hidden_channels + group * dv + c) = value;
          }
        }
        __syncwarp();
      }
    }
    __syncthreads();  // Output projection consumes all groups.

    for (int col = warp_id * 32; col < p.channels; col += Warps * 32) {
      Fragment accum; accum.clear();
      for (int kc = 0; kc < hidden_channels; kc += 32) {
        for (int e = lane * Access; e < 16 * 32; e += 32 * Access) {
          int row = e / 32, k = e % 32;
          copy(s.a.data() + la({row, k}), hidden + row * hidden_channels + kc + k, kc + k < hidden_channels);
        }
        for (int e = lane * Access; e < 32 * 32; e += 32 * Access) {
          int n = e / 32, k = e % 32;
          copy(s.b.data() + lb({k, n}), args.output_weight + int64_t(col + n) * hidden_channels + kc + k,
               col + n < p.channels && kc + k < hidden_channels);
        }
        mma_step(s, accum, lane);
      }
      store(s, accum, lane);
      for (int e = lane * Access; e < 16 * 32; e += 32 * Access) {
        int row = e / 32, c = col + e % 32;
        if (c < p.channels && (window_output || scatter[row] >= 0)) {
          AccessType value;
          #pragma unroll
          for (int v = 0; v < Access; ++v) {
            float bias = args.output_bias ? float(args.output_bias[c + v]) : 0;
            value[v] = Element(s.accum.data()[e + v] + bias);
          }
          Element* dst = window_output ? window_output + row * p.channels + c
              : args.output + int64_t(scatter[row]) * p.channels + c;
          *reinterpret_cast<AccessType*>(dst) = value;
        }
      }
      __syncwarp();
    }
  }
};

}  // namespace tiny_cutlass::swin::window_attention::threadblock
