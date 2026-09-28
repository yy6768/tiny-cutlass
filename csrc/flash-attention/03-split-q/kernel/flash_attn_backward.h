#pragma once

#include "cutlass/cutlass.h"
#include "../../flash_attention.h"

template <class Mma_, class Epilogue_, class ArchTag_>
struct FlashAttnBackwardKernel {
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using ArchTag = ArchTag_;
  using CanonicalIterator = typename Mma::CanonicalIterator;
  using CanonicalParams = typename CanonicalIterator::Params;
  using SwizzleIterator = typename Mma::SwizzleIterator;
  using SwizzleParams = typename SwizzleIterator::Params;
  using SharedStorage = typename Mma::SharedStorage;
  using FragmentGrad = typename Mma::FragmentGrad;
  using OutputTileIterator = typename Epilogue::OutputTileIterator;
  static constexpr int kThreadCount = Mma::kThreadCount;
  static constexpr int kHeadDim = Mma::kHeadDim;
  static constexpr int kBr = Mma::kBr;
  static constexpr int kBc = Mma::kBc;

  struct Params {
    Problem problem;
    BackwardTensors tensors;
    float const* delta = nullptr;
    float* grad_query_accum = nullptr;
    CanonicalParams canonical;
    SwizzleParams swizzle;
    Params() = default;
    Params(Problem const& p, BackwardTensors const& t, float const* d, float* dq)
        : problem(p), tensors(t), delta(d), grad_query_accum(dq),
          canonical(cutlass::layout::PitchLinear(p.head_number * p.head_size)),
          swizzle(cutlass::layout::PitchLinear(p.head_number * p.head_size)) {}
  };

  CUTLASS_DEVICE
  void operator()(Params const& params, SharedStorage& storage) {
    auto const& p = params.problem;
    auto const& t = params.tensors;
    int key_start = int(blockIdx.x) * kBc;
    int head = int(blockIdx.y);
    int batch = int(blockIdx.z);
    if (key_start >= p.seq_length_kv || head >= p.head_number || batch >= p.batch_size)
      return;

    int thread = int(threadIdx.x);
    int warp = cutlass::canonical_warp_idx_sync();
    int lane = thread % 32;
    int valid_keys = min(kBc, p.seq_length_kv - key_start);
    int query_iterations = (p.seq_length - 1) / kBr + 1;
    int last_query_rows = (p.seq_length - 1) % kBr + 1;
    int padded_query_rows = query_iterations * kBr;
    int64_t stride = int64_t(p.head_number) * p.head_size;
    int64_t query_base = int64_t(batch) * p.seq_length * stride + int64_t(head) * p.head_size;
    int64_t key_base = int64_t(batch) * p.seq_length_kv * stride + int64_t(head) * p.head_size;
    int64_t row_base = (int64_t(batch) * p.head_number + head) * p.seq_length;
    int copy_thread = thread % Mma::kCanonicalCopyThreads;
    CanonicalIterator query(params.canonical, t.query + query_base,
        {p.head_size, padded_query_rows}, copy_thread, {0, 0});
    CanonicalIterator key(params.canonical, t.key + key_base,
        {p.head_size, key_start + valid_keys}, copy_thread, {0, key_start});
    CanonicalIterator value(params.canonical, t.value + key_base,
        {p.head_size, key_start + valid_keys}, copy_thread, {0, key_start});
    CanonicalIterator grad_output(params.canonical, t.grad_output + query_base,
        {p.head_size, padded_query_rows}, copy_thread, {0, 0});
    SwizzleIterator query_swizzle(params.swizzle, t.query + query_base,
        {p.head_size, padded_query_rows}, thread, {0, 0});
    SwizzleIterator key_swizzle(params.swizzle, t.key + key_base,
        {p.head_size, key_start + valid_keys}, thread, {0, key_start});
    SwizzleIterator grad_output_swizzle(params.swizzle, t.grad_output + query_base,
        {p.head_size, padded_query_rows}, thread, {0, 0});

    FragmentGrad grad_key, grad_value;
    grad_key.clear();
    grad_value.clear();
    Mma mma(storage, thread, warp, lane);
    mma(query_iterations, last_query_rows, valid_keys, grad_key, grad_value,
        query, key, value, grad_output, query_swizzle, key_swizzle, grad_output_swizzle,
        p.scale, t.logsumexp + row_base, params.delta + row_base,
        params.grad_query_accum + query_base, stride);

    // Each KV CTA and warp owns disjoint dK/dV rows. Physical channel padding
    // never changes the public output extent or global tensor stride.
    int warp_row = warp * 16;
    int valid_rows = max(0, min(16, valid_keys - warp_row));
    int64_t output_offset = key_base + int64_t(key_start + warp_row) * stride;
    OutputTileIterator output_k(
        {t.grad_key + output_offset, cutlass::layout::RowMajor(stride)},
        {valid_rows, p.head_size}, lane);
    OutputTileIterator output_v(
        {t.grad_value + output_offset, cutlass::layout::RowMajor(stride)},
        {valid_rows, p.head_size}, lane);
    Epilogue epilogue;
    epilogue(grad_key, output_k);
    epilogue(grad_value, output_v);
  }
};
