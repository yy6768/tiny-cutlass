#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "../../flash_attention.h"

// Coordinates and composition only. Attention arithmetic lives in Mma/Epilogue.
template <class Mma_, class Epilogue_, class ArchTag_>
struct FlashAttnKernel {
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using ArchTag = ArchTag_;
  using IteratorQ = typename Mma::IteratorQ;
  using IteratorK = typename Mma::IteratorK;
  using IteratorV = typename Mma::IteratorV;
  using SharedStorage = typename Mma::SharedStorage;
  static constexpr int kThreadCount = Mma::kThreadCount;
  static constexpr int kBr = Mma::kBr;
  static constexpr int kBc = Mma::kBc;
  static constexpr int kHeadDim = Mma::kHeadDim;
  static constexpr int kHeadDimV = Mma::kHeadDimV;
  struct Params {
    Problem problem;
    Tensors tensors;
    typename IteratorQ::Params params_q;
    typename IteratorK::Params params_k;
    typename IteratorV::Params params_v;

    Params() = default;

    // Precompute global iterator strides on the host. A single head is a
    // [sequence, channel] view of BSHD with row stride H * D.
    Params(Problem const& p, Tensors const& t)
        : problem(p), tensors(t),
          params_q(cutlass::layout::PitchLinear(p.head_number * p.head_size)),
          params_k(cutlass::layout::PitchLinear(p.head_number * p.head_size)),
          params_v(cutlass::layout::PitchLinear(p.head_number * p.head_size_v)) {}
  };

  CUTLASS_DEVICE
  void operator()(Params const& params, SharedStorage& storage) {
    // Compute ThreadBlock location

    auto const& p = params.problem;
    auto const& t = params.tensors;

    // Identity CTA mapping: (Q tile, head, batch). Exit collectively before
    // constructing pointers or reaching any warp/CTA synchronization.
    int q_tile = int(blockIdx.x); 
    int head = int(blockIdx.y);
    int batch = int(blockIdx.z);
    if (q_tile >= (p.seq_length - 1) / kBr + 1 ||
        head >= p.head_number || batch >= p.batch_size) {
      return;
    }

    // Compute initial location in logical [sequence, channel] coordinates.
    cutlass::MatrixCoord tb_offset_q(q_tile * kBr, 0);
    cutlass::MatrixCoord tb_offset_k(0, 0);
    cutlass::MatrixCoord tb_offset_v(0, 0);
    int valid_q = min(kBr, p.seq_length - tb_offset_q.row());
    int64_t stride_qk = int64_t(p.head_number) * p.head_size;
    int64_t stride_vo = int64_t(p.head_number) * p.head_size_v;
    int64_t offset_q = int64_t(batch) * p.seq_length * stride_qk + int64_t(head) * p.head_size;
    int64_t offset_k = int64_t(batch) * p.seq_length_kv * stride_qk + int64_t(head) * p.head_size;
    int64_t offset_v = int64_t(batch) * p.seq_length_kv * stride_vo + int64_t(head) * p.head_size_v;

    // The outer attention loop scans KV sequence tiles, not head-dimension K.
    // PredicatedTileAccessIterator visits the residue tile first.
    int kv_tile_iterations = (p.seq_length_kv - 1) / kBc + 1;
    int first_kv_rows = (p.seq_length_kv - 1) % kBc + 1;

    // Construct global operand iterators. PitchLinear coordinates are
    // (channel, sequence); K's transposed shared view belongs to the loader.
    int thread_idx = int(threadIdx.x);
    IteratorQ iterator_q(params.params_q, t.query + offset_q,
        {p.head_size, tb_offset_q.row() + valid_q}, thread_idx,
        {tb_offset_q.column(), tb_offset_q.row()});
    IteratorK iterator_k(params.params_k, t.key + offset_k,
        {p.head_size, p.seq_length_kv}, thread_idx,
        {tb_offset_k.column(), tb_offset_k.row()});
    IteratorV iterator_v(params.params_v, t.value + offset_v,
        {p.head_size_v, p.seq_length_kv}, thread_idx,
        {tb_offset_v.column(), tb_offset_v.row()});

    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = thread_idx % 32;

    // Mainloop: shared/warp iterators and stage ownership live inside Mma.
    typename Mma::FragmentO accum;
    typename Mma::FragmentL denominator;
    accum.clear();
    denominator.clear();
    Mma mma(storage, thread_idx, warp_idx, lane_idx);
    mma(kv_tile_iterations, first_kv_rows, accum, denominator,
        iterator_q, iterator_k, iterator_v, p.scale);

    // Epilogue: locate this CTA's output tile, normalize and predicate stores.
    // This epilogue currently consumes a TensorRef, not a stock output iterator.
    cutlass::MatrixCoord tb_offset_o = tb_offset_q;
    int64_t offset_o = (int64_t(batch) * p.seq_length + tb_offset_o.row()) * stride_vo
        + int64_t(head) * p.head_size_v + tb_offset_o.column();
    typename Epilogue::TensorRefO ref_o(t.output + offset_o,
        cutlass::layout::RowMajor(stride_vo));
    Epilogue epilogue;
    epilogue(accum, denominator, ref_o, valid_q, warp_idx, lane_idx);
  }
};
