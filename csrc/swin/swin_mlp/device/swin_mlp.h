#pragma once

/*
  Device-level driver for the Swin MLP:

      y = x + fc2(GELU(fc1(LayerNorm(x)) + b1)) + b2

  Three launches: LN2, fc1 (+bias +GELU in its epilogue), fc2 (+bias +residual
  in its epilogue). See kernel/default_swin_mlp.h for why LN2 cannot fold into
  fc1 and why fc1/fc2 stay separate in this step.
*/

#include <cstddef>

#include <cuda_runtime_api.h>

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/half.h"

#include "swin/swin_mlp/kernel/default_swin_mlp.h"
#include "swin/swin_problem.h"

namespace tiny_cutlass::swin::swin_mlp::device {

template <
    typename ArchTag_ = cutlass::arch::Sm80,
    typename Element_ = cutlass::half_t,
    typename ElementCompute_ = float>
class SwinMlp {
 public:
  using ArchTag = ArchTag_;
  using Element = Element_;
  using ElementCompute = ElementCompute_;

  using KernelConfig =
      kernel::DefaultSwinMlp<ArchTag, Element, float, ElementCompute>;
  using LayerNorm = typename KernelConfig::LayerNorm;
  using Fc1 = typename KernelConfig::Fc1;
  using Fc2 = typename KernelConfig::Fc2;

  /// Arguments. `rows` is the token count; the operator is layout-agnostic about
  /// whether those tokens are in image order or window order, so the same
  /// operator serves any stage.
  ///
  /// Weights are [out_features, in_features] row-major -- the layout
  /// nn.Linear.weight uses -- read as column-major operand B.
  struct Arguments {
    int rows = 0;
    int channels = 0;
    int hidden = 0;

    Element const* input = nullptr;   // [rows, channels], also the residual
    ElementCompute const* gamma = nullptr;  // LN2 scale, [channels]
    ElementCompute const* beta = nullptr;   // LN2 shift, [channels]

    Element const* fc1_weight = nullptr;    // [hidden, channels]
    ElementCompute const* fc1_bias = nullptr;  // [hidden]
    Element const* fc2_weight = nullptr;    // [channels, hidden]
    ElementCompute const* fc2_bias = nullptr;  // [channels]

    Element* output = nullptr;        // [rows, channels]

    // Workspace: normalized tokens [rows, channels] and the activated hidden
    // tensor [rows, hidden]. Both are caller-provided so the operator does no
    // allocation.
    Element* normalized = nullptr;
    Element* hidden_buffer = nullptr;

    ElementCompute epsilon = ElementCompute(1e-5f);
  };

  static cutlass::Status can_implement(Arguments const& args) {
    if (args.input == nullptr || args.output == nullptr ||
        args.fc1_weight == nullptr || args.fc2_weight == nullptr ||
        args.normalized == nullptr || args.hidden_buffer == nullptr) {
      return cutlass::Status::kErrorInvalidProblem;
    }
    if (args.rows <= 0 || args.channels <= 0 || args.hidden <= 0) {
      return cutlass::Status::kErrorInvalidProblem;
    }
    // Vectorized alignment on every GEMM operand and on the LayerNorm.
    if (args.channels % KernelConfig::kAlignment != 0 ||
        args.hidden % KernelConfig::kAlignment != 0) {
      return cutlass::Status::kErrorNotSupported;
    }
    return cutlass::Status::kSuccess;
  }

  /// Bytes of workspace the caller must supply for `normalized`.
  static size_t normalized_size(int rows, int channels) {
    return size_t(rows) * size_t(channels) * sizeof(Element);
  }

  /// Bytes of workspace the caller must supply for `hidden_buffer`.
  static size_t hidden_size(int rows, int hidden) {
    return size_t(rows) * size_t(hidden) * sizeof(Element);
  }

  cutlass::Status run(Arguments const& args, cudaStream_t stream = nullptr) {
    cutlass::Status status = can_implement(args);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    //
    // 1. LN2 over the channel axis.
    //
    typename LayerNorm::Arguments ln_args;
    ln_args.input = args.input;
    ln_args.output = args.normalized;
    ln_args.gamma = args.gamma;
    ln_args.beta = args.beta;
    ln_args.rows = args.rows;
    ln_args.cols = args.channels;
    ln_args.epsilon = args.epsilon;

    status = LayerNorm::run(ln_args, stream);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    //
    // 2. fc1: [rows, C] x [C, hidden], + b1, then GELU.
    //
    // The epilogue computes z = binary_op(alpha*accum + beta*C, V) followed by
    // the activation. With beta = 0 and V bound to the bias vector that is
    // GELU(accum + b1) -- so no source tensor is needed.
    typename Fc1::Arguments fc1_args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        cutlass::gemm::GemmCoord{args.rows, args.hidden, args.channels},
        /*batch_count=*/1,
        {ElementCompute(1.0f), ElementCompute(0.0f)},
        args.normalized,
        args.fc1_weight,
        /*ptr_C=*/nullptr,   // beta = 0, so no source tensor is read
        args.hidden_buffer,
        /*ptr_Vector=*/const_cast<ElementCompute*>(args.fc1_bias),
        /*ptr_Tensor=*/nullptr,
        /*batch_stride_A=*/int64_t(0),
        /*batch_stride_B=*/int64_t(0),
        /*batch_stride_C=*/int64_t(0),
        /*batch_stride_D=*/int64_t(0),
        /*batch_stride_Vector=*/int64_t(0),
        /*batch_stride_Tensor=*/int64_t(0),
        /*lda=*/args.channels,
        /*ldb=*/args.channels,  // column-major [hidden, channels]
        /*ldc=*/0,
        /*ldd=*/args.hidden,
        /*ldr=*/0,   // bias vector is broadcast, so its row stride is 0
        /*ldt=*/0);

    Fc1 fc1;
    status = fc1.can_implement(fc1_args);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }
    status = fc1.initialize(fc1_args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }
    status = fc1(stream);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    //
    // 3. fc2: [rows, hidden] x [hidden, C], + b2 + residual.
    //
    // beta = 1 with the source tensor bound to the block input makes the
    // epilogue compute (accum + residual) + b2 in one pass. b2 is added ONCE
    // here, not per hidden tile -- there is no hidden-tile loop at this level,
    // which is exactly the trap a manual tiling would fall into.
    typename Fc2::Arguments fc2_args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        cutlass::gemm::GemmCoord{args.rows, args.channels, args.hidden},
        /*batch_count=*/1,
        {ElementCompute(1.0f), ElementCompute(1.0f)},
        args.hidden_buffer,
        args.fc2_weight,
        /*ptr_C=*/args.input,   // residual2
        args.output,
        /*ptr_Vector=*/const_cast<ElementCompute*>(args.fc2_bias),
        /*ptr_Tensor=*/nullptr,
        /*batch_stride_A=*/int64_t(0),
        /*batch_stride_B=*/int64_t(0),
        /*batch_stride_C=*/int64_t(0),
        /*batch_stride_D=*/int64_t(0),
        /*batch_stride_Vector=*/int64_t(0),
        /*batch_stride_Tensor=*/int64_t(0),
        /*lda=*/args.hidden,
        /*ldb=*/args.hidden,  // column-major [channels, hidden]
        /*ldc=*/args.channels,
        /*ldd=*/args.channels,
        /*ldr=*/0,
        /*ldt=*/0);

    Fc2 fc2;
    status = fc2.can_implement(fc2_args);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }
    status = fc2.initialize(fc2_args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }
    return fc2(stream);
  }

  cutlass::Status operator()(
      Arguments const& args, cudaStream_t stream = nullptr) {
    return run(args, stream);
  }
};

}  // namespace tiny_cutlass::swin::swin_mlp::device
