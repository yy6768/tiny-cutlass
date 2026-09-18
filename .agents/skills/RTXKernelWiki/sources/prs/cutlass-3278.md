---
id: pr-cutlass-3278
title: Fix the ScatterD issue in predicated_tile_iterator
type: pr
url: https://github.com/NVIDIA/cutlass/pull/3278
repo: NVIDIA/cutlass
number: 3278
architectures:
- sm89
architecture_evidence:
  sm89: load_with_byte_offset 的 group/cluster 增量增加 !ScatterD 条件，operator+= 与 operator++
    对齐。
tags:
- cutlass-2x
- gather-scatter
- iterator
kernel_types:
- gemm
- attention
symptoms:
- incorrect-output
- indexing
status: merged
confidence: inferred
retrieved_at: '2026-09-18'
updated_at: '2026-08-18T01:20:28Z'
inclusion_reason: 修正 2.x PredicatedTileIterator 的 ScatterD 指针推进，供 partition/reverse
  组件审查。
local_validation: not-run
head_sha: dbea1dcc4979c88353d661f898a6e2f0b047ba4e
merged_at: '2026-08-18T01:20:28Z'
merge_commit_sha: 1d7cde107fa2edbf8fbf7460498b5b36a0cf8f0a
changed_files: 1
code_urls:
- https://github.com/NVIDIA/cutlass/blob/dbea1dcc4979c88353d661f898a6e2f0b047ba4e/include/cutlass/epilogue/threadblock/predicated_tile_iterator.h
evidence_api: https://api.github.com/repos/NVIDIA/cutlass/pulls/3278
research_focus: sm89-local-attention
candidate_role: indexing-component
priority: P1
---

# Fix the ScatterD issue in predicated_tile_iterator

[上游 PR](https://github.com/NVIDIA/cutlass/pull/3278) · merged `2026-08-18T01:20:28Z` · 核查 2026-09-18

## 对本任务的价值

**P1 / indexing-component**。修正 2.x PredicatedTileIterator 的 ScatterD 指针推进，供 partition/reverse 组件审查。

load_with_byte_offset 的 group/cluster 增量增加 !ScatterD 条件，operator+= 与 operator++ 对齐。

## 适用边界

作者的复现是 SIMT INT8 GEMM，不是 SM89 FP8 TensorOp attention。本地 fused attention 当前直接消费 host 行索引；只有实际使用该 ScatterD iterator 的路径才受此修复影响。

收录代表值得研究，不代表采用。GPU parity/性能：`not-run`。

## 固定版本实现

Head：`dbea1dcc4979c88353d661f898a6e2f0b047ba4e`。PR 共 1 个变更文件；已抓取 1 个关键源码文件。

- [include/cutlass/epilogue/threadblock/predicated_tile_iterator.h](../../artifacts/github/NVIDIA--cutlass/pr-3278/20260917T165052550363Z/key-files/head/include/cutlass/epilogue/threadblock/predicated_tile_iterator.h) · [固定 SHA 上游](https://github.com/NVIDIA/cutlass/blob/dbea1dcc4979c88353d661f898a6e2f0b047ba4e/include/cutlass/epilogue/threadblock/predicated_tile_iterator.h)

[正文、讨论、完整 diff、来源清单和其余源码](../../wiki/candidates/pr-NVIDIA--cutlass-3278.md)。自动 Wiki 保持未审核标记，人工筛选结论以本页为准。
