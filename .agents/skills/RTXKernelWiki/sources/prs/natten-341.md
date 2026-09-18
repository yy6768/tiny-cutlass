---
id: pr-natten-341
title: Fix/cutlass fna dv nan issue
type: pr
url: https://github.com/SHI-Labs/NATTEN/pull/341
repo: SHI-Labs/NATTEN
number: 341
architectures:
- sm89
architecture_evidence:
  sm89: kernel_backward.h 删除 __CUDA_ARCH__ < 800 限定，使无效 query/key accumulator 都参与边界屏蔽。
tags:
- fna
- mask
- backward
- partial-tile
kernel_types:
- attention
symptoms:
- nan
- boundary-semantics
status: merged
confidence: inferred
retrieved_at: '2026-09-18'
updated_at: '2026-06-24T14:37:20Z'
inclusion_reason: 修复 FNA backward 在 SM80+ partial tile 上遗漏 mask 边界检查而出现 dV NaN。
local_validation: not-run
head_sha: e1ed3b64087d2f230f692bc48249ef804c46bfa9
merged_at: '2026-06-24T14:36:58Z'
merge_commit_sha: e4177231fa4921b6582d940494734bb50cc67e2a
changed_files: 3
code_urls:
- https://github.com/SHI-Labs/NATTEN/blob/e1ed3b64087d2f230f692bc48249ef804c46bfa9/csrc/include/natten/cuda/fna/kernel_backward.h
evidence_api: https://api.github.com/repos/SHI-Labs/NATTEN/pulls/341
research_focus: sm89-local-attention
candidate_role: backward-reference
priority: P2
---

# Fix/cutlass fna dv nan issue

[上游 PR](https://github.com/SHI-Labs/NATTEN/pull/341) · merged `2026-06-24T14:36:58Z` · 核查 2026-09-18

## 对本任务的价值

**P2 / backward-reference**。修复 FNA backward 在 SM80+ partial tile 上遗漏 mask 边界检查而出现 dV NaN。

kernel_backward.h 删除 __CUDA_ARCH__ < 800 限定，使无效 query/key accumulator 都参与边界屏蔽。

## 适用边界

本地当前仅做 forward；这条作为边界测试与未来 backward 的证据，不能说它修复了当前 forward。没有 SM89 FP8 backward 结论。

收录代表值得研究，不代表采用。GPU parity/性能：`not-run`。

## 固定版本实现

Head：`e1ed3b64087d2f230f692bc48249ef804c46bfa9`。PR 共 3 个变更文件；已抓取 2 个关键源码文件。

- [csrc/include/natten/cuda/fna/kernel_backward.h](../../artifacts/github/SHI-Labs--NATTEN/pr-341/20260917T165018000073Z/key-files/head/csrc/include/natten/cuda/fna/kernel_backward.h) · [固定 SHA 上游](https://github.com/SHI-Labs/NATTEN/blob/e1ed3b64087d2f230f692bc48249ef804c46bfa9/csrc/include/natten/cuda/fna/kernel_backward.h)

[正文、讨论、完整 diff、来源清单和其余源码](../../wiki/candidates/pr-SHI-Labs--NATTEN-341.md)。自动 Wiki 保持未审核标记，人工筛选结论以本页为准。
