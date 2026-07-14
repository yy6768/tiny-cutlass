/*
  Device-level launch helpers for Swin threadblock stages.
*/

#pragma once

#include <cuda_runtime.h>

#include "kernel/swin.h"

namespace tiny_cutlass {
namespace swin {
namespace device {
namespace detail {

template <typename Threadblock>
cudaError_t launch_threadblock(
    typename Threadblock::Params const& params,
    cudaStream_t stream,
    int shared_storage_bytes = 0) {
  kernel::swin_threadblock_kernel<Threadblock>
      <<<params.getBlocksGrid(),
         params.getThreadsGrid(),
         shared_storage_bytes,
         stream>>>(params);
  return cudaGetLastError();
}

} // namespace detail
} // namespace device
} // namespace swin
} // namespace tiny_cutlass
