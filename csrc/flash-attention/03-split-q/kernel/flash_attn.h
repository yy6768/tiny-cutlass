#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "../../flash_attention.h"

// One CTA owns one Q/output tile. Mma owns the complete KV traversal.
template <class Mma_, class Epilogue_, class ArchTag_>
struct FlashAttnKernelSplitQ {
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using ArchTag = ArchTag_;
  using IteratorQ = typename Mma::IteratorQ;
  using IteratorK = typename Mma::IteratorK;
  using IteratorV = typename Mma::IteratorV;
  using IteratorParamsQ = typename IteratorQ::Params;
  using IteratorParamsK = typename IteratorK::Params;
  using IteratorParamsV = typename IteratorV::Params;
  using FragmentO = typename Mma::FragmentO;
  using FragmentL = typename Mma::FragmentL;
  using WarpShapeQK = typename Mma::WarpShapeQK;
  using OutputTileIterator = typename Epilogue::OutputTileIterator;
  using SharedStorage = typename Mma::SharedStorage;
  static constexpr int kThreadCount = Mma::kThreadCount;
  static constexpr int kBr = Mma::kBr;
  static constexpr int kBc = Mma::kBc;
  static constexpr int kHeadDim = Mma::kHeadDim;
  static constexpr int kHeadDimV = Mma::kHeadDimV;

  struct Params {
    Problem problem;
    Tensors tensors;
    IteratorParamsQ params_q;
    IteratorParamsK params_k;
    IteratorParamsV params_v;
    int q_tile_count = 0;

    Params() = default;
    Params(Problem const& p, Tensors const& t)
        : problem(p), tensors(t),
          params_q(cutlass::layout::PitchLinear(p.head_number * p.head_size)),
          params_k(cutlass::layout::PitchLinear(p.head_number * p.head_size)),
          params_v(cutlass::layout::PitchLinear(p.head_number * p.head_size_v)),
          q_tile_count((p.seq_length - 1) / kBr + 1) {}
  };

  CUTLASS_DEVICE
  void operator()(Params const& params, SharedStorage& storage) {
    auto const& p = params.problem;
    auto const& t = params.tensors;

    // Identity CTA mapping: (Q tile, head, batch). Early exit is CTA-uniform.
    int q_tile = int(blockIdx.x);
    int head = int(blockIdx.y);
    int batch = int(blockIdx.z);
    if (q_tile >= params.q_tile_count || head >= p.head_number || batch >= p.batch_size)
      return;

    // Compute initial tile locations within the selected batch/head.
    cutlass::MatrixCoord tb_offset_q(q_tile * kBr, 0);
    cutlass::MatrixCoord tb_offset_k(0, 0);
    cutlass::MatrixCoord tb_offset_v(0, 0);
    int valid_q = min(kBr, p.seq_length - tb_offset_q.row());
    int64_t stride_qk = int64_t(p.head_number) * p.head_size;
    int64_t stride_vo = int64_t(p.head_number) * p.head_size_v;
    int64_t offset_q = int64_t(batch) * p.seq_length * stride_qk + int64_t(head) * p.head_size;
    int64_t offset_k = int64_t(batch) * p.seq_length_kv * stride_qk + int64_t(head) * p.head_size;
    int64_t offset_v = int64_t(batch) * p.seq_length_kv * stride_vo + int64_t(head) * p.head_size_v;

    // Count KV tiles. A padded iterator extent visits complete tiles first;
    // the QK head-dimension loop and PV reduction loop belong to Mma/warp.
    int kv_tile_iterations = (p.seq_length_kv - 1) / kBc + 1;
    int last_kv_rows = (p.seq_length_kv - 1) % kBc + 1;
    int padded_kv_rows = kv_tile_iterations * kBc;
    int thread_idx = int(threadIdx.x);
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = thread_idx % 32;

    // Construct global iterators once. PitchLinear coordinates are (D, sequence).
    IteratorQ iterator_q(params.params_q, t.query + offset_q,
        {p.head_size, tb_offset_q.row() + valid_q}, lane_idx,
        {tb_offset_q.column(), tb_offset_q.row()});
    IteratorK iterator_k(params.params_k, t.key + offset_k,
        {p.head_size, padded_kv_rows}, lane_idx,
        {tb_offset_k.column(), tb_offset_k.row()});
    IteratorV iterator_v(params.params_v, t.value + offset_v,
        {p.head_size_v, padded_kv_rows}, thread_idx % Mma::kCopyThreadsV,
        {tb_offset_v.column(), tb_offset_v.row()});

    // Mainloop: one fixed Q tile, all KV tiles, FP32 online state in registers.
    FragmentO accum;
    FragmentL denominator;
    FragmentL logsumexp;
    accum.clear();
    denominator.clear();
    Mma mma(storage, thread_idx, warp_idx, lane_idx);
    mma(kv_tile_iterations, last_kv_rows, accum, denominator,
        iterator_q, iterator_k, iterator_v, p.scale, logsumexp, t.logsumexp != nullptr);

    // Epilogue: normalize and write this CTA's output tile once.
    cutlass::MatrixCoord tb_offset_o = tb_offset_q;
    int warp_row = warp_idx * WarpShapeQK::kM;
    int64_t offset_o = (int64_t(batch) * p.seq_length + tb_offset_o.row() + warp_row) * stride_vo
        + int64_t(head) * p.head_size_v;
    OutputTileIterator output(
        {t.output + offset_o, cutlass::layout::RowMajor(stride_vo)},
        {max(0, min(WarpShapeQK::kM, valid_q - warp_row)), p.head_size_v}, lane_idx);
    Epilogue epilogue;
    epilogue(accum, denominator, output);
    int64_t offset_l = (int64_t(batch) * p.head_number + head) * p.seq_length
        + tb_offset_q.row() + warp_row;
    epilogue.store_logsumexp(logsumexp,
        t.logsumexp ? t.logsumexp + offset_l : nullptr,
        max(0, min(WarpShapeQK::kM, valid_q - warp_row)), lane_idx);
  }
};
