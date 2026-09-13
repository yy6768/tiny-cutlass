#pragma once

/*
  Policy factory for the Swin MLP:

      y = x + fc2(GELU(fc1(LayerNorm(x)) + b1)) + b2

  ---------------------------------------------------------------------------
  STAGE STRUCTURE AND WHY IT IS NOT ONE KERNEL
  ---------------------------------------------------------------------------
  Three pieces, in dependency order:

    LN2   token LayerNorm over C -- a SEPARATE pass, because it normalizes
          along what is fc1's K axis. See
          swin/layernorm/kernel/token_layernorm.h for the full argument; the
          short version is that a CTA has not seen a whole row when operand A is
          loaded, so the statistics do not exist yet. CUTLASS's own
          GemmLayernormMainloopFusion also takes mean/var precomputed.

    fc1   [tokens, C] x [C, hidden] + b1, then GELU. Bias and activation are
          folded into the epilogue via LinearCombinationBiasElementwise, so the
          hidden tensor is written exactly once, already activated.

    fc2   [tokens, hidden] x [hidden, C] + b2 + residual. Both the bias
          (per-column broadcast) and the residual (full tensor) land in one
          epilogue using LinearCombinationBiasElementwise with its source tensor
          bound to the residual.

  fc1 and fc2 stay separate launches in this step. Fusing them means staging
  fc1's accumulator through shared memory so fc2 reads operand A from smem
  (the from-smem B2B pattern); that is a fusion step, and this round's goal is
  independent, verified families first.

  ---------------------------------------------------------------------------
  GELU MUST BE THE EXACT ERF FORM
  ---------------------------------------------------------------------------
  cutlass::epilogue::thread::GELU uses erf and matches the host reference's
  std::erf. The tanh approximation (GELU_taylor) differs by ~1e-3, which is
  large enough to read as a real defect in a parity test. Do not swap it.
*/

#include "cutlass/arch/mma.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_bias_elementwise.h"
#include "cutlass/gemm/device/gemm_universal_with_broadcast.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"

#include "swin/layernorm/device/token_layernorm.h"

namespace tiny_cutlass::swin::swin_mlp::kernel {

template <
    typename ArchTag_,
    typename Element_,
    typename ElementAccumulator_ = float,
    typename ElementCompute_ = float,
    typename ThreadblockShape_ = cutlass::gemm::GemmShape<128, 128, 32>,
    typename WarpShape_ = cutlass::gemm::GemmShape<64, 64, 32>,
    typename InstructionShape_ = cutlass::gemm::GemmShape<16, 8, 16>,
    int Stages = 3>
struct DefaultSwinMlp {
  using ArchTag = ArchTag_;
  using Element = Element_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  static int const kStages = Stages;
  static int const kAlignment = 8;

  // Tokens are row-major [rows, channels]. Weights are stored so that the GEMM
  // reads them column-major, i.e. the host holds [out_features, in_features]
  // row-major -- the same layout PyTorch's nn.Linear.weight uses.
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using LayerNorm = layernorm::device::TokenLayerNorm<Element, ElementCompute>;

  // The epilogue computes  z = BinaryOp(alpha*accum + beta*C, V)  and then
  // applies ElementwiseOp, where V is the broadcast bias vector. With
  // BinaryOp = plus that gives exactly what both layers need:
  //
  //   fc1 (beta = 0)         : GELU(accum + b1)
  //   fc2 (beta = 1, C = x)  : Identity((accum + residual) + b2)
  //
  // Parameter order is (..., ElementwiseOp, BinaryOp, StoreT, ElementVector):
  // StoreT is the NINTH parameter, after BinaryOp. ElementVector must be spelled
  // out as ElementCompute because it defaults to ElementC (fp16) while the bias
  // tensors here are fp32.
  static int const kEpilogueElementsPerAccess =
      128 / cutlass::sizeof_bits<Element>::value;

  // fc1: StoreT = false because the pre-activation tensor is only needed for a
  // backward pass, which this family does not implement.
  using Fc1EpilogueOp = cutlass::epilogue::thread::LinearCombinationBiasElementwise<
      Element,             // ElementC
      ElementAccumulator,  // ElementAccumulator
      ElementCompute,      // ElementCompute
      Element,             // ElementZ (the stored output)
      Element,             // ElementT (unused when StoreT = false)
      kEpilogueElementsPerAccess,
      cutlass::epilogue::thread::GELU<ElementCompute>,
      cutlass::plus<ElementCompute>,
      /*StoreT=*/false,
      ElementCompute>;     // ElementVector: fp32 bias

  using Fc2EpilogueOp = cutlass::epilogue::thread::LinearCombinationBiasElementwise<
      Element,
      ElementAccumulator,
      ElementCompute,
      Element,
      Element,
      kEpilogueElementsPerAccess,
      cutlass::epilogue::thread::Identity<ElementCompute>,
      cutlass::plus<ElementCompute>,
      /*StoreT=*/false,
      ElementCompute>;

  using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using Fc1 = cutlass::gemm::device::GemmUniversalWithBroadcast<
      Element, LayoutA,
      Element, LayoutB,
      Element, LayoutC,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      ArchTag,
      ThreadblockShape, WarpShape, InstructionShape,
      Fc1EpilogueOp,
      Swizzle,
      kStages,
      kAlignment, kAlignment>;

  using Fc2 = cutlass::gemm::device::GemmUniversalWithBroadcast<
      Element, LayoutA,
      Element, LayoutB,
      Element, LayoutC,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      ArchTag,
      ThreadblockShape, WarpShape, InstructionShape,
      Fc2EpilogueOp,
      Swizzle,
      kStages,
      kAlignment, kAlignment>;
};

}  // namespace tiny_cutlass::swin::swin_mlp::kernel
