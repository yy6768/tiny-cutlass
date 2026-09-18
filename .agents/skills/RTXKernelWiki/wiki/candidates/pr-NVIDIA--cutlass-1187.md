---
id: candidate-pr-NVIDIA--cutlass-1187
title: fix cp.async L2 prefetch typo
type: pr
repo: NVIDIA/cutlass
architectures:
- sm80
- sm86
- sm89
status: merged
tags: []
review: defer
url: https://github.com/NVIDIA/cutlass/pull/1187
inclusion_reason: 自动抓取 candidate；架构来自搜索命中，需阅读代码确认。
local_validation: not-run
retrieved_at: 20260917T151849564241Z
artifact_dir: artifacts/github/NVIDIA--cutlass/pr-1187/20260917T151849564241Z
---

# fix cp.async L2 prefetch typo

候选状态：**defer / 未审核**。架构标签是发现线索，不是设备支持证明。

- [GitHub](https://github.com/NVIDIA/cutlass/pull/1187)
- [原始正文与讨论](../../artifacts/github/NVIDIA--cutlass/pr-1187/20260917T151849564241Z/page.md)
- [来源清单](../../artifacts/github/NVIDIA--cutlass/pr-1187/20260917T151849564241Z/PROVENANCE.yaml)
- [完整 PR diff](../../artifacts/github/NVIDIA--cutlass/pr-1187/20260917T151849564241Z/diff.patch)

Head SHA: `21424b0c5fa2bd31e8fa5655ee4252b7a01953d1` · merged

## 关键实现文件

- [include/cute/arch/copy_sm80.hpp](../../artifacts/github/NVIDIA--cutlass/pr-1187/20260917T151849564241Z/key-files/head/include/cute/arch/copy_sm80.hpp)（modified，`21424b0c5fa2`）
