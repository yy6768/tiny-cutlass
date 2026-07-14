/*
  Kernel-level launch wrappers for Swin threadblock stages.
*/

#pragma once

namespace tiny_cutlass {
namespace swin {
namespace kernel {

template <typename Threadblock>
__global__ void __launch_bounds__(Threadblock::kNumThreads, Threadblock::kMinBlocksPerSm)
    swin_threadblock_kernel(typename Threadblock::Params params) {
  Threadblock::run(params);
}

} // namespace kernel
} // namespace swin
} // namespace tiny_cutlass
