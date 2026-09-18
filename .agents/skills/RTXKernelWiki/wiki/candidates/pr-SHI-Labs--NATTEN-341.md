---
id: candidate-pr-SHI-Labs--NATTEN-341
title: Fix/cutlass fna dv nan issue
type: pr
repo: SHI-Labs/NATTEN
architectures:
- sm89
status: merged
tags: []
review: defer
url: https://github.com/SHI-Labs/NATTEN/pull/341
inclusion_reason: 自动抓取 candidate；架构来自搜索命中，需阅读代码确认。
local_validation: not-run
retrieved_at: 20260917T165018000073Z
artifact_dir: artifacts/github/SHI-Labs--NATTEN/pr-341/20260917T165018000073Z
---

# Fix/cutlass fna dv nan issue

候选状态：**defer / 未审核**。架构标签是发现线索，不是设备支持证明。

- [GitHub](https://github.com/SHI-Labs/NATTEN/pull/341)
- [原始正文与讨论](../../artifacts/github/SHI-Labs--NATTEN/pr-341/20260917T165018000073Z/page.md)
- [来源清单](../../artifacts/github/SHI-Labs--NATTEN/pr-341/20260917T165018000073Z/PROVENANCE.yaml)
- [完整 PR diff](../../artifacts/github/SHI-Labs--NATTEN/pr-341/20260917T165018000073Z/diff.patch)

Head SHA: `e1ed3b64087d2f230f692bc48249ef804c46bfa9` · merged

## 关键实现文件

- [csrc/include/natten/cuda/fna/kernel_backward.h](../../artifacts/github/SHI-Labs--NATTEN/pr-341/20260917T165018000073Z/key-files/head/csrc/include/natten/cuda/fna/kernel_backward.h)（modified，`e1ed3b64087d`）
- [src/natten/version.py](../../artifacts/github/SHI-Labs--NATTEN/pr-341/20260917T165018000073Z/key-files/head/src/natten/version.py)（modified，`e1ed3b64087d`）
