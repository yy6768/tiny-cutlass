---
id: pr-natten-337
title: 'cutlass-fna: 64b strides'
type: pr
url: https://github.com/SHI-Labs/NATTEN/pull/337
repo: SHI-Labs/NATTEN
number: 337
architectures:
- sm89
architecture_evidence:
  sm89: kernel_forward.h 的 q/k/v/o/lse stride 从 Dim 改为独立 Stride，na_utils.cuh 和多处 iterator
    同步修改；属于可迁移的寻址修正。
tags:
- fna
- indexing
- stride
- cutlass-2x
kernel_types:
- attention
symptoms:
- integer-overflow
- large-shapes
status: merged
confidence: inferred
retrieved_at: '2026-09-18'
updated_at: '2026-05-10T02:33:42Z'
inclusion_reason: FNA token strides 改为 64 位；固定 head 保存较新的 forward kernel 和邻域索引实现。
local_validation: not-run
head_sha: 6fd34669017a39cd37b10d9edd3c65c88c18dfb6
merged_at: '2026-05-10T02:33:38Z'
merge_commit_sha: 92750c3cf837652d58b6091e5ebd37dcad46e753
changed_files: 11
code_urls:
- https://github.com/SHI-Labs/NATTEN/blob/6fd34669017a39cd37b10d9edd3c65c88c18dfb6/csrc/include/natten/cuda/fna/kernel_forward.h
- https://github.com/SHI-Labs/NATTEN/blob/6fd34669017a39cd37b10d9edd3c65c88c18dfb6/csrc/include/natten/cuda/fna/na_utils.cuh
evidence_api: https://api.github.com/repos/SHI-Labs/NATTEN/pulls/337
research_focus: sm89-local-attention
candidate_role: correctness-reference
priority: P1
---

# cutlass-fna: 64b strides

[上游 PR](https://github.com/SHI-Labs/NATTEN/pull/337) · merged `2026-05-10T02:33:38Z` · 核查 2026-09-18

## 对本任务的价值

**P1 / correctness-reference**。FNA token strides 改为 64 位；固定 head 保存较新的 forward kernel 和邻域索引实现。

kernel_forward.h 的 q/k/v/o/lse stride 从 Dim 改为独立 Stride，na_utils.cuh 和多处 iterator 同步修改；属于可迁移的寻址修正。

## 适用边界

大 token 数或 head dimension 时审查乘法中间值与步长溢出。它不新增 FP8，也不改变本地现有 1D、stride=dilation=1 契约；64 位寻址的实际开销需单独测量。

收录代表值得研究，不代表采用。GPU parity/性能：`not-run`。

## 固定版本实现

Head：`6fd34669017a39cd37b10d9edd3c65c88c18dfb6`。PR 共 11 个变更文件；已抓取 10 个关键源码文件。

- [csrc/include/natten/cuda/fna/kernel_forward.h](../../artifacts/github/SHI-Labs--NATTEN/pr-337/20260917T164959673620Z/key-files/head/csrc/include/natten/cuda/fna/kernel_forward.h) · [固定 SHA 上游](https://github.com/SHI-Labs/NATTEN/blob/6fd34669017a39cd37b10d9edd3c65c88c18dfb6/csrc/include/natten/cuda/fna/kernel_forward.h)
- [csrc/include/natten/cuda/fna/na_utils.cuh](../../artifacts/github/SHI-Labs--NATTEN/pr-337/20260917T164959673620Z/key-files/head/csrc/include/natten/cuda/fna/na_utils.cuh) · [固定 SHA 上游](https://github.com/SHI-Labs/NATTEN/blob/6fd34669017a39cd37b10d9edd3c65c88c18dfb6/csrc/include/natten/cuda/fna/na_utils.cuh)

[正文、讨论、完整 diff、来源清单和其余源码](../../wiki/candidates/pr-SHI-Labs--NATTEN-337.md)。自动 Wiki 保持未审核标记，人工筛选结论以本页为准。
