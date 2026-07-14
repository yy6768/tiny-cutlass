/*
  Device-level Swin attention path.
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
class SwinAttention {
 public:
  using ArchTag = ArchTag_;
  using Element = Element_;
  using Tensors = SwinAttentionTensors<Element>;

  struct Arguments {
    SwinAttentionProblem problem;
    Tensors tensors;
  };

  static cutlass::Status can_implement(SwinAttentionProblem const& problem);

  static cutlass::Status can_implement(Arguments const& args) {
    return can_implement(args.problem);
  }

  static cutlass::Status run(
      SwinAttentionProblem const& problem,
      Tensors const& tensors,
      cudaStream_t stream);

  static cutlass::Status run(Arguments const& args, cudaStream_t stream) {
    return run(args.problem, args.tensors, stream);
  }
};

} // namespace device
} // namespace swin
} // namespace tiny_cutlass
