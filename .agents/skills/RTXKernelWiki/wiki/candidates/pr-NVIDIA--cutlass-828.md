---
id: candidate-pr-NVIDIA--cutlass-828
title: 'fMHA: Sync FW with xFormers'
type: pr
repo: NVIDIA/cutlass
architectures:
- sm89
status: merged
tags: []
review: defer
url: https://github.com/NVIDIA/cutlass/pull/828
inclusion_reason: 自动抓取 candidate；架构来自搜索命中，需阅读代码确认。
local_validation: not-run
retrieved_at: 20260917T165023228733Z
artifact_dir: artifacts/github/NVIDIA--cutlass/pr-828/20260917T165023228733Z
---

# fMHA: Sync FW with xFormers

候选状态：**defer / 未审核**。架构标签是发现线索，不是设备支持证明。

- [GitHub](https://github.com/NVIDIA/cutlass/pull/828)
- [原始正文与讨论](../../artifacts/github/NVIDIA--cutlass/pr-828/20260917T165023228733Z/page.md)
- [来源清单](../../artifacts/github/NVIDIA--cutlass/pr-828/20260917T165023228733Z/PROVENANCE.yaml)
- [完整 PR diff](../../artifacts/github/NVIDIA--cutlass/pr-828/20260917T165023228733Z/diff.patch)

Head SHA: `8b0bb170c1c4c8e7d97d73871399d43de32be938` · merged

## 关键实现文件

- [examples/41_fused_multi_head_attention/debug_utils.h](../../artifacts/github/NVIDIA--cutlass/pr-828/20260917T165023228733Z/key-files/head/examples/41_fused_multi_head_attention/debug_utils.h)（modified，`8b0bb170c1c4`）
- [examples/41_fused_multi_head_attention/default_fmha_grouped.h](../../artifacts/github/NVIDIA--cutlass/pr-828/20260917T165023228733Z/key-files/head/examples/41_fused_multi_head_attention/default_fmha_grouped.h)（modified，`8b0bb170c1c4`）
- [examples/41_fused_multi_head_attention/epilogue/epilogue_pipelined.h](../../artifacts/github/NVIDIA--cutlass/pr-828/20260917T165023228733Z/key-files/head/examples/41_fused_multi_head_attention/epilogue/epilogue_pipelined.h)（renamed，`8b0bb170c1c4`）
- [examples/41_fused_multi_head_attention/epilogue/epilogue_rescale_output.h](../../artifacts/github/NVIDIA--cutlass/pr-828/20260917T165023228733Z/key-files/head/examples/41_fused_multi_head_attention/epilogue/epilogue_rescale_output.h)（renamed，`8b0bb170c1c4`）
- [examples/41_fused_multi_head_attention/epilogue/epilogue_thread_apply_logsumexp.h](../../artifacts/github/NVIDIA--cutlass/pr-828/20260917T165023228733Z/key-files/head/examples/41_fused_multi_head_attention/epilogue/epilogue_thread_apply_logsumexp.h)（renamed，`8b0bb170c1c4`）
- [examples/41_fused_multi_head_attention/fmha_grouped.h](../../artifacts/github/NVIDIA--cutlass/pr-828/20260917T165023228733Z/key-files/head/examples/41_fused_multi_head_attention/fmha_grouped.h)（modified，`8b0bb170c1c4`）
- [examples/41_fused_multi_head_attention/fused_multihead_attention_fixed_seqlen.cu](../../artifacts/github/NVIDIA--cutlass/pr-828/20260917T165023228733Z/key-files/head/examples/41_fused_multi_head_attention/fused_multihead_attention_fixed_seqlen.cu)（modified，`8b0bb170c1c4`）
- [examples/41_fused_multi_head_attention/gemm/find_default_mma.h](../../artifacts/github/NVIDIA--cutlass/pr-828/20260917T165023228733Z/key-files/head/examples/41_fused_multi_head_attention/gemm/find_default_mma.h)（renamed，`8b0bb170c1c4`）
- [examples/41_fused_multi_head_attention/gemm/mma_accum_lambda_iterator.h](../../artifacts/github/NVIDIA--cutlass/pr-828/20260917T165023228733Z/key-files/head/examples/41_fused_multi_head_attention/gemm/mma_accum_lambda_iterator.h)（renamed，`8b0bb170c1c4`）
- [examples/41_fused_multi_head_attention/gemm/mma_from_smem.h](../../artifacts/github/NVIDIA--cutlass/pr-828/20260917T165023228733Z/key-files/head/examples/41_fused_multi_head_attention/gemm/mma_from_smem.h)（renamed，`8b0bb170c1c4`）
- [examples/41_fused_multi_head_attention/kernel_forward.h](../../artifacts/github/NVIDIA--cutlass/pr-828/20260917T165023228733Z/key-files/head/examples/41_fused_multi_head_attention/kernel_forward.h)（modified，`8b0bb170c1c4`）
- [examples/41_fused_multi_head_attention/transform/tile_smem_loader.h](../../artifacts/github/NVIDIA--cutlass/pr-828/20260917T165023228733Z/key-files/head/examples/41_fused_multi_head_attention/transform/tile_smem_loader.h)（added，`8b0bb170c1c4`）
