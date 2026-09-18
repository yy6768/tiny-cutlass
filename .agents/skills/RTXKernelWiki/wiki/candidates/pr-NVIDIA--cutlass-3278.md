---
id: candidate-pr-NVIDIA--cutlass-3278
title: Fix the ScatterD issue in predicated_tile_iterator
type: pr
repo: NVIDIA/cutlass
architectures:
- sm89
status: merged
tags: []
review: defer
url: https://github.com/NVIDIA/cutlass/pull/3278
inclusion_reason: 自动抓取 candidate；架构来自搜索命中，需阅读代码确认。
local_validation: not-run
retrieved_at: 20260917T165052550363Z
artifact_dir: artifacts/github/NVIDIA--cutlass/pr-3278/20260917T165052550363Z
---

# Fix the ScatterD issue in predicated_tile_iterator

候选状态：**defer / 未审核**。架构标签是发现线索，不是设备支持证明。

- [GitHub](https://github.com/NVIDIA/cutlass/pull/3278)
- [原始正文与讨论](../../artifacts/github/NVIDIA--cutlass/pr-3278/20260917T165052550363Z/page.md)
- [来源清单](../../artifacts/github/NVIDIA--cutlass/pr-3278/20260917T165052550363Z/PROVENANCE.yaml)
- [完整 PR diff](../../artifacts/github/NVIDIA--cutlass/pr-3278/20260917T165052550363Z/diff.patch)

Head SHA: `dbea1dcc4979c88353d661f898a6e2f0b047ba4e` · merged

## 关键实现文件

- [include/cutlass/epilogue/threadblock/predicated_tile_iterator.h](../../artifacts/github/NVIDIA--cutlass/pr-3278/20260917T165052550363Z/key-files/head/include/cutlass/epilogue/threadblock/predicated_tile_iterator.h)（modified，`dbea1dcc4979`）
