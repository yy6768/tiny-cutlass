---
id: pr-cutlass-828
title: 'fMHA: Sync FW with xFormers'
type: pr
url: https://github.com/NVIDIA/cutlass/pull/828
repo: NVIDIA/cutlass
number: 828
architectures:
- sm89
architecture_evidence:
  sm89: kernel_forward.h、gemm/mma_from_smem.h 和 epilogue_rescale_output.h 是 CUTLASS
    2.x 风格融合实现；对 SM89 的适用性属于源码推断。
tags:
- cutlass-2x
- online-softmax
- attention-bias
- fusion
kernel_types:
- attention
symptoms:
- bias-scale-order
- intermediate-memory
status: merged
confidence: inferred
retrieved_at: '2026-09-18'
updated_at: '2023-02-23T04:25:32Z'
inclusion_reason: CUTLASS example 41 的 online softmax、attention bias 和 shared-memory
  PV 组件。
local_validation: not-run
head_sha: 8b0bb170c1c4c8e7d97d73871399d43de32be938
merged_at: '2023-02-23T04:25:31Z'
merge_commit_sha: f303889ed9d296633af1d606125f39488c128f6f
changed_files: 12
code_urls:
- https://github.com/NVIDIA/cutlass/blob/8b0bb170c1c4c8e7d97d73871399d43de32be938/examples/41_fused_multi_head_attention/kernel_forward.h
- https://github.com/NVIDIA/cutlass/blob/8b0bb170c1c4c8e7d97d73871399d43de32be938/examples/41_fused_multi_head_attention/gemm/mma_from_smem.h
- https://github.com/NVIDIA/cutlass/blob/8b0bb170c1c4c8e7d97d73871399d43de32be938/examples/41_fused_multi_head_attention/epilogue/epilogue_rescale_output.h
evidence_api: https://api.github.com/repos/NVIDIA/cutlass/pulls/828
research_focus: sm89-local-attention
candidate_role: attention-mainline
priority: P0
---

# fMHA: Sync FW with xFormers

[上游 PR](https://github.com/NVIDIA/cutlass/pull/828) · merged `2023-02-23T04:25:31Z` · 核查 2026-09-18

## 对本任务的价值

**P0 / attention-mainline**。CUTLASS example 41 的 online softmax、attention bias 和 shared-memory PV 组件。

kernel_forward.h、gemm/mma_from_smem.h 和 epilogue_rescale_output.h 是 CUTLASS 2.x 风格融合实现；对 SM89 的适用性属于源码推断。

## 适用边界

有 bias 时，该 head 先将 QK 乘 scale，再加 bias（kernel_forward.h:765-796）；本地约定是 (QK+bias)*scale，移植必须保持本地 reference。上游并未完成本地 reflect/RMSNorm/QKV/proj 全链，也未提供本任务 FP8 契约。

收录代表值得研究，不代表采用。GPU parity/性能：`not-run`。

## 固定版本实现

Head：`8b0bb170c1c4c8e7d97d73871399d43de32be938`。PR 共 12 个变更文件；已抓取 12 个关键源码文件。

- [examples/41_fused_multi_head_attention/kernel_forward.h](../../artifacts/github/NVIDIA--cutlass/pr-828/20260917T165023228733Z/key-files/head/examples/41_fused_multi_head_attention/kernel_forward.h) · [固定 SHA 上游](https://github.com/NVIDIA/cutlass/blob/8b0bb170c1c4c8e7d97d73871399d43de32be938/examples/41_fused_multi_head_attention/kernel_forward.h)
- [examples/41_fused_multi_head_attention/gemm/mma_from_smem.h](../../artifacts/github/NVIDIA--cutlass/pr-828/20260917T165023228733Z/key-files/head/examples/41_fused_multi_head_attention/gemm/mma_from_smem.h) · [固定 SHA 上游](https://github.com/NVIDIA/cutlass/blob/8b0bb170c1c4c8e7d97d73871399d43de32be938/examples/41_fused_multi_head_attention/gemm/mma_from_smem.h)
- [examples/41_fused_multi_head_attention/epilogue/epilogue_rescale_output.h](../../artifacts/github/NVIDIA--cutlass/pr-828/20260917T165023228733Z/key-files/head/examples/41_fused_multi_head_attention/epilogue/epilogue_rescale_output.h) · [固定 SHA 上游](https://github.com/NVIDIA/cutlass/blob/8b0bb170c1c4c8e7d97d73871399d43de32be938/examples/41_fused_multi_head_attention/epilogue/epilogue_rescale_output.h)

[正文、讨论、完整 diff、来源清单和其余源码](../../wiki/candidates/pr-NVIDIA--cutlass-828.md)。自动 Wiki 保持未审核标记，人工筛选结论以本页为准。
