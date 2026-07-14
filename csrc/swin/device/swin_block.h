/*
  Device-level fused Swin block facade (declaration).

  Implements the full Microsoft SwinTransformerBlock (v1 pre-norm) as a single
  device::SwinBlock<ArchTag, Element>::run call. The launch sequence fuses glue kernels
  FT-style to minimize kernel launch count:

  Launch #  | Kernel                          | Fuses
  ----------|---------------------------------|------------------------------
  1         | LayerNormShiftPartition         | norm1 + shift + partition
  2         | GEMM (QKV projection)           | C -> 3C
  3         | AddQkvBiasSplit                 | bias + split Q/K/V
  4         | WindowAttentionCore             | flash-attn (online softmax)
  5         | GEMM (output projection)        | C -> C
  6         | ReverseAddResidualLayerNorm     | reverse + residual1 + norm2
  7         | GEMM (fc1)                      | C -> 4C
  8         | AddBiasGelu                     | fc1 bias + GELU
  9         | GEMM (fc2)                      | 4C -> C
  10        | AddBiasResidual                 | fc2 bias + residual2

  Total: 10 launches for a complete SwinBlock. The definition lives in swin.cu
  (with explicit instantiation) so the CUDA-only attention headers stay out of
  the public interface, matching device::SwinAttention.
*/

#pragma once

#include <cuda_runtime.h>

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "swin_problem.h"

namespace tiny_cutlass {
namespace swin {
namespace device {

template <typename ArchTag_, typename Element_>
class SwinBlock {
 public:
  using ArchTag = ArchTag_;
  using Element = Element_;
  using Tensors = SwinBlockTensors<Element>;

  struct Arguments {
    SwinBlockProblem problem;
    Tensors tensors;
  };

  static cutlass::Status can_implement(SwinBlockProblem const& problem);

  static cutlass::Status can_implement(Arguments const& args) {
    return can_implement(args.problem);
  }

  static cutlass::Status run(
      SwinBlockProblem const& problem,
      Tensors const& tensors,
      cudaStream_t stream);

  static cutlass::Status run(Arguments const& args, cudaStream_t stream) {
    return run(args.problem, args.tensors, stream);
  }
};

} // namespace device
} // namespace swin
} // namespace tiny_cutlass
