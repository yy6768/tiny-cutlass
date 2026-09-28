#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/layout/matrix.h>
#include "natten/nb_atten.h"
#include "natten/neighborhood.h"

namespace tiny_cutlass { namespace natten {

// CTA position, problem-tail checks, initial iterators and composition only.
// The Q/K/V traversal and shared memory synchronization live in Mma.
template <class Mma_, class Epilogue_, class ArchTag_>
struct NeighborhoodKernel {
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using ArchTag = ArchTag_;
  using Element = typename Mma::Element;
  using IteratorQ = typename Mma::IteratorQ;
  using IteratorK = typename Mma::IteratorK;
  using IteratorV = typename Mma::IteratorV;
  using OutputTileIterator = typename Epilogue::OutputTileIterator;
  using SharedStorage = typename Mma::SharedStorage;
  static constexpr int kBr = Mma::kBr;
  static constexpr int kBc = Mma::kBc;
  static constexpr int kThreads = Mma::kThreads;

  struct Params {
    NeighborhoodProblem problem;
    Element const* query = nullptr;
    Element const* key = nullptr;
    Element const* value = nullptr;
    Element* output = nullptr;
    float* lse = nullptr;
    typename IteratorQ::Params q_params;
    typename IteratorK::Params k_params;
    typename IteratorV::Params v_params;

    Params() = default;
    Params(NeighborhoodProblem const& p, Element const* q, Element const* k,
           Element const* v, Element* o, float* l)
        : problem(p), query(q), key(k), value(v), output(o), lse(l),
          q_params(cutlass::layout::PitchLinear(p.heads * p.head_dim)),
          k_params(cutlass::layout::PitchLinear(p.heads * p.head_dim)),
          v_params(cutlass::layout::PitchLinear(p.heads * p.head_dim_value)) {}
  };

  CUTLASS_DEVICE void operator()(Params const& params, SharedStorage& storage) {
    auto const& p = params.problem;
    int tile = int(blockIdx.x);
    int head = int(blockIdx.y);
    int batch = int(blockIdx.z);
    int query_base = tile * kBr;
    if (query_base >= p.length || head >= p.heads || batch >= p.batch_size) return;

    int last_query = query_base + kBr - 1;
    last_query = last_query < p.length ? last_query : p.length - 1;
    int key_begin = neighborhood_start(query_base, p.length, p.kernel_size);
    int key_end = neighborhood_start(last_query, p.length, p.kernel_size) + p.kernel_size;
    int key_tile_count = (key_end - key_begin + kBc - 1) / kBc;
    int last_key_rows = (key_end - key_begin - 1) % kBc + 1;
    int64_t qk_stride = int64_t(p.heads) * p.head_dim;
    int64_t vo_stride = int64_t(p.heads) * p.head_dim_value;
    int64_t q_offset = int64_t(batch) * p.length * qk_stride +
        int64_t(query_base) * qk_stride + int64_t(head) * p.head_dim;
    int64_t k_offset = int64_t(batch) * p.length * qk_stride +
        int64_t(key_begin) * qk_stride + int64_t(head) * p.head_dim;
    int64_t v_offset = int64_t(batch) * p.length * vo_stride +
        int64_t(key_begin) * vo_stride + int64_t(head) * p.head_dim_value;
    int64_t o_offset = int64_t(batch) * p.length * vo_stride +
        int64_t(query_base) * vo_stride + int64_t(head) * p.head_dim_value;
    int rows = key_end - key_begin;
    int padded_rows = ((rows + kBc - 1) / kBc) * kBc;
    int thread = int(threadIdx.x);
    int warp = cutlass::canonical_warp_idx_sync();
    int lane = thread % 32;
    IteratorQ q(params.q_params, params.query + q_offset,
                {p.head_dim, kBr}, thread);
    IteratorK k(params.k_params, params.key + k_offset,
                {p.head_dim, padded_rows}, thread);
    IteratorV v(params.v_params, params.value + v_offset,
                {p.head_dim_value, padded_rows}, thread);
    int warp_row = warp * Mma::WarpQK::Shape::kM;
    OutputTileIterator out(
        {params.output + o_offset + int64_t(warp_row) * vo_stride,
         cutlass::layout::RowMajor(vo_stride)},
        {p.length - query_base - warp_row, p.head_dim_value}, lane);
    float* lse = params.lse
        ? params.lse + (int64_t(batch) * p.heads + head) * ((p.length + 31) / 32 * 32)
        : nullptr;
    Mma mma(storage, thread, warp, lane);
    Epilogue epilogue;
    mma(query_base, key_begin, key_tile_count, last_key_rows,
        p.length, p.kernel_size, p.scale,
        q, k, v, out, lse, epilogue);
  }
};

}} // namespace tiny_cutlass::natten
