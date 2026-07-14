/*
  Device launch helpers for Swin window partition and reverse stages.
*/

#pragma once

#include <cuda_runtime.h>

#include "device/launch.h"
#include "swin_problem.h"
#include "threadblock/window_partition.h"

namespace tiny_cutlass {
namespace swin {
namespace device {
namespace detail {

template <typename Element, int kThreads>
cudaError_t launch_window_partition(
    SwinAttentionProblem const& problem,
    SwinAttentionTensors<Element> const& tensors,
    cudaStream_t stream) {
  using Threadblock = threadblock::WindowPartition<Element, kThreads>;
  typename Threadblock::Params params;
  params.input = tensors.input;
  params.output = tensors.attention.windows;
  params.batch = problem.batch_size;
  params.height = problem.image_size;
  params.width = problem.image_size;
  params.channels = swin_channels(problem);
  params.shift_size = problem.shift_size;
  params.window_size = problem.window_size;
  return launch_threadblock<Threadblock>(params, stream);
}

template <typename Element, int kThreads>
cudaError_t launch_window_reverse(
    SwinAttentionProblem const& problem,
    SwinAttentionTensors<Element> const& tensors,
    cudaStream_t stream) {
  using Threadblock = threadblock::WindowReverse<Element, kThreads>;
  typename Threadblock::Params params;
  params.input = tensors.attention.projected;
  params.output = tensors.output;
  params.batch = problem.batch_size;
  params.height = problem.image_size;
  params.width = problem.image_size;
  params.channels = swin_channels(problem);
  params.shift_size = problem.shift_size;
  params.window_size = problem.window_size;
  return launch_threadblock<Threadblock>(params, stream);
}

} // namespace detail

} // namespace device
} // namespace swin
} // namespace tiny_cutlass
