#pragma once

namespace tiny_cutlass::swin::fused_swin_layer::threadblock {

template <typename Mma_, typename Epilogue_, bool SharedB>
struct Gemm {
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using Element = typename Mma::IteratorA::Element;
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  CUTLASS_DEVICE
  void operator()(SharedStorage& shared, Element* a, int lda,
                  Element const* b, int ldb, int k, float* d) const {
    int tid = int(threadIdx.x);
    typename Mma::IteratorA iterator_A({a, cutlass::layout::RowMajor(lda)}, tid);
    typename Mma::FragmentC accum; accum.clear();
    Mma mma(shared.main_loop, tid, 0, tid);
    if constexpr (SharedB) {
      typename Mma::IteratorB iterator_B(
          {const_cast<Element*>(b), typename Mma::IteratorB::Layout(ldb)}, tid);
      mma(k / 32, accum, iterator_A, iterator_B, accum);
    } else {
      typename Mma::IteratorB::Params params{typename Mma::IteratorB::Layout(ldb)};
      typename Mma::IteratorB iterator_B(params, const_cast<Element*>(b), {k, 32}, tid);
      mma(k / 32, accum, iterator_A, iterator_B, accum);
    }
    __syncthreads();
    using Iterator = typename Epilogue::OutputTileIterator;
    typename Iterator::Params params{cutlass::layout::RowMajor(32)};
    Iterator destination(params, d, {16, 32}, tid);
    Epilogue epilogue(shared.epilogue, tid, 0, tid);
    typename Epilogue::OutputOp output_op(typename Epilogue::OutputOp::Params(1, 0));
    epilogue(output_op, destination, accum);
    __syncthreads();
  }
};

}  // namespace tiny_cutlass::swin::fused_swin_layer::threadblock
