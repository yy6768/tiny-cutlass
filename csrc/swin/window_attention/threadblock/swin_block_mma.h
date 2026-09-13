#pragma once
#include "cutlass/epilogue/thread/activation.h"
#include "swin/window_attention/ops/swin_block.h"

namespace tiny_cutlass::swin::window_attention::threadblock {

template <typename AttentionMma_, int Channels, int Ratio, bool StaticAttention = false>
struct SwinBlockMma {
  using AttentionMma = AttentionMma_;
  using Element = typename AttentionMma::Element;
  using AccessType = typename AttentionMma::AccessType;
  static constexpr int Access = AttentionMma::kAccess;
  using Fragment = typename AttentionMma::Fragment;
  using Params = SwinBlockArguments<Element>;
  static constexpr int kWarps = AttentionMma::kWarps;
  static constexpr int kMaxQk = Channels, kMaxValue = Channels;
  static constexpr int kHidden = Channels * Ratio;
  static_assert(Channels == 32 && Ratio == 4, "this block policy requires C=32, ratio=4");

  struct OperandStorage {
    cutlass::AlignedBuffer<Element, 16 * 32> a;
    cutlass::AlignedBuffer<Element, 32 * 32> b;
    cutlass::AlignedBuffer<float, 16 * 32> accum;
  };
  struct SharedStorage {
    union {
      struct {
        typename AttentionMma::SharedStorage storage;
        cutlass::AlignedBuffer<Element, 16 * (Channels + Channels)> scratch;
      } attention;
      struct {
        OperandStorage warp[kWarps];
        cutlass::AlignedBuffer<Element, 16 * Channels> normalized;
        cutlass::AlignedBuffer<Element, 16 * kHidden> hidden;
      } mlp;
    };
    cutlass::AlignedBuffer<Element, 16 * Channels> residual;
  };

  CUTLASS_DEVICE
  void operator()(Params const& args, SharedStorage& shared, int window) const {
    int lane = int(threadIdx.x) % 32, warp_id = int(threadIdx.x) / 32;
    AttentionMma mma;
    WindowAttentionArguments<Element> attention_args = args;
    if constexpr (StaticAttention) {
      attention_args.problem.channels = Channels;
      attention_args.problem.groups = 1;
      attention_args.problem.qk_channels = Channels;
      attention_args.problem.value_channels = Channels;
    }
    mma(attention_args, shared.attention.storage, window, shared.residual.data());
    __syncthreads();  // All attention users finish before reusing its storage.

    int const* gather = args.gather + int64_t(window) * 16;
    int const* scatter = args.scatter + int64_t(window) * 16;
    if constexpr (Access == 8) {
      int c = lane % 4 * 8;
      AccessType gamma = *reinterpret_cast<AccessType const*>(args.mlp_rms_weight + c);
      for (int row = warp_id * 8 + lane / 4; row < 16; row += kWarps * 8) {
        AccessType r = *reinterpret_cast<AccessType const*>(shared.residual.data() + row * Channels + c);
        AccessType x = *reinterpret_cast<AccessType const*>(args.input + int64_t(gather[row]) * Channels + c);
        float sum = 0;
        #pragma unroll
        for (int v = 0; v < 8; ++v) {
          r[v] = Element(float(r[v]) + float(x[v]));
          sum += float(r[v]) * float(r[v]);
        }
        *reinterpret_cast<AccessType*>(shared.residual.data() + row * Channels + c) = r;
        sum += __shfl_xor_sync(0xffffffff, sum, 1, 4);
        sum += __shfl_xor_sync(0xffffffff, sum, 2, 4);
        float inv = rsqrtf(sum / Channels + args.mlp_rms_epsilon);
        #pragma unroll
        for (int v = 0; v < 8; ++v) r[v] = Element(float(r[v]) * inv * float(gamma[v]));
        *reinterpret_cast<AccessType*>(shared.mlp.normalized.data() + row * Channels + c) = r;
      }
    } else {
    for (int row = warp_id; row < 16; row += kWarps) {
      float r = float(Element(float(shared.residual.data()[row * Channels + lane]) +
                              float(args.input[int64_t(gather[row]) * Channels + lane])));
      shared.residual.data()[row * Channels + lane] = Element(r);
      float sum = r * r;
      #pragma unroll
      for (int delta = 16; delta; delta /= 2)
        sum += __shfl_xor_sync(0xffffffff, sum, delta);
      float inv = rsqrtf(sum / Channels + args.mlp_rms_epsilon);
      shared.mlp.normalized.data()[row * Channels + lane] = Element(r * inv * float(args.mlp_rms_weight[lane]));
    }
    }
    __syncthreads();
    auto& s = shared.mlp.warp[warp_id];
    typename AttentionMma::LayoutA la(32);
    typename AttentionMma::LayoutB lb(32);
    for (int col = warp_id * 32; col < kHidden; col += kWarps * 32) {
      Fragment accum; accum.clear();
      for (int e = lane * Access; e < 16 * 32; e += 32 * Access)
        mma.copy(s.a.data() + la({e / 32, e % 32}), shared.mlp.normalized.data() + e);
      for (int e = lane * Access; e < 32 * 32; e += 32 * Access)
        mma.copy(s.b.data() + lb({e % 32, e / 32}), args.fc1_weight + (col + e / 32) * Channels + e % 32);
      mma.mma_step(s, accum, lane);
      mma.store(s, accum, lane);
      for (int e = lane * Access; e < 16 * 32; e += 32 * Access) {
        // Match eager Linear's half rounding before the separate exact GELU.
        AccessType value;
        #pragma unroll
        for (int v = 0; v < Access; ++v) {
          float z = float(Element(s.accum.data()[e + v] + float(args.fc1_bias[col + e % 32 + v])));
          value[v] = Element(cutlass::epilogue::thread::GELU<float>{}(z));
        }
        *reinterpret_cast<AccessType*>(shared.mlp.hidden.data() + (e / 32) * kHidden + col + e % 32) = value;
      }
      __syncwarp();
    }
    __syncthreads();
    if (warp_id == 0) {
      Fragment accum; accum.clear();
      #pragma unroll
      for (int kc = 0; kc < kHidden; kc += 32) {
        for (int e = lane * Access; e < 16 * 32; e += 32 * Access)
          mma.copy(s.a.data() + la({e / 32, e % 32}), shared.mlp.hidden.data() + (e / 32) * kHidden + kc + e % 32);
        for (int e = lane * Access; e < 32 * 32; e += 32 * Access)
          mma.copy(s.b.data() + lb({e % 32, e / 32}), args.fc2_weight + (e / 32) * kHidden + kc + e % 32);
        mma.mma_step(s, accum, lane);
      }
      mma.store(s, accum, lane);
      for (int e = lane * Access; e < 16 * 32; e += 32 * Access) {
        int row = e / 32, c = e % 32;
        if (scatter[row] >= 0) {
          AccessType value;
          #pragma unroll
          for (int v = 0; v < Access; ++v) {
            float projected = float(Element(s.accum.data()[e + v] + float(args.fc2_bias[c + v])));
            value[v] = Element(projected + float(shared.residual.data()[e + v]));
          }
          *reinterpret_cast<AccessType*>(args.output + int64_t(scatter[row]) * Channels + c) = value;
        }
      }
    }
  }
};

}  // namespace tiny_cutlass::swin::window_attention::threadblock
