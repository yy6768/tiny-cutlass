/*
  Device-level Swin operation facade.
*/

#pragma once

#include <cuda_runtime.h>

#include "../swin_problem.h"

namespace tiny_cutlass {
namespace swin {
namespace device {

template <typename Kernel_>
class Swin {
 public:
  using Kernel = Kernel_;
  using Policy = typename Kernel::Policy;
  using Element = typename Kernel::Element;
  using Tensors = SwinTensors<Element>;

  struct Arguments {
    SwinProblem problem;
    Tensors tensors;
  };

  static bool can_implement(SwinProblem const& problem, char const** reason);

  static bool can_implement(Arguments const& args, char const** reason) {
    return can_implement(args.problem, reason);
  }

  static cudaError_t run(
      SwinProblem const& problem,
      Tensors const& tensors,
      cudaStream_t stream);

  static cudaError_t run(Arguments const& args, cudaStream_t stream) {
    return run(args.problem, args.tensors, stream);
  }
};

} // namespace device
} // namespace swin
} // namespace tiny_cutlass
