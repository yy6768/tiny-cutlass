/*
  Device-level Swin PatchEmbed operation facade.
*/

#pragma once

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"

#include "../swin_problem.h"

namespace tiny_cutlass {
namespace swin {
namespace device {

template <typename Kernel_>
class PatchEmbed {
 public:
  using Kernel = Kernel_;
  using Policy = typename Kernel::Policy;
  using Element = typename Kernel::Element;
  using Tensors = PatchEmbedTensors<Element>;

  struct Arguments {
    PatchEmbedProblem problem;
    Tensors tensors;
  };

  static bool can_implement(PatchEmbedProblem const& problem, char const** reason);

  static bool can_implement(Arguments const& args, char const** reason) {
    return can_implement(args.problem, reason);
  }

  static cutlass::Status run(
      PatchEmbedProblem const& problem,
      Tensors const& tensors,
      cudaStream_t stream);

  static cutlass::Status run(Arguments const& args, cudaStream_t stream) {
    return run(args.problem, args.tensors, stream);
  }
};

} // namespace device
} // namespace swin
} // namespace tiny_cutlass
