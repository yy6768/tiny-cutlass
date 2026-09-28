#pragma once
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/epilogue/thread/linear_combination.h"

namespace tiny_cutlass::swin::fused_swin_layer::epilogue {

template <typename ThreadblockShape, typename WarpMma>
struct DefaultEpilogue {
  // Keep accumulations in float until the explicit eager-half operation boundary.
  using OutputOp = cutlass::epilogue::thread::LinearCombination<float, 4, float, float>;
  using Default = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
      ThreadblockShape, WarpMma, 1, OutputOp, OutputOp::kCount>::Epilogue;
  // CUTLASS's generic CUDA store supports a shared destination. beta=0, so the
  // global-only source-C load is never called. All other epilogue types are stock.
  using OutputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
      typename Default::OutputTileIterator::ThreadMap, float, false,
      cutlass::layout::NoPermute, true>;
  using Epilogue = cutlass::epilogue::threadblock::Epilogue<
      ThreadblockShape, WarpMma, 1, OutputTileIterator,
      typename Default::AccumulatorFragmentIterator, typename Default::WarpTileIterator,
      typename Default::SharedLoadIterator, OutputOp, typename Default::Padding>;
};

}  // namespace tiny_cutlass::swin::fused_swin_layer::epilogue
