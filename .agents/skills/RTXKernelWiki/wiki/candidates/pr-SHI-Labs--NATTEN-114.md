---
id: candidate-pr-SHI-Labs--NATTEN-114
title: Disable 64x128x128 GEMM config for SM86 and 89
type: pr
repo: SHI-Labs/NATTEN
architectures:
- sm89
status: merged
tags: []
review: defer
url: https://github.com/SHI-Labs/NATTEN/pull/114
inclusion_reason: 自动抓取 candidate；架构来自搜索命中，需阅读代码确认。
local_validation: not-run
retrieved_at: 20260917T164959670620Z
artifact_dir: artifacts/github/SHI-Labs--NATTEN/pr-114/20260917T164959670620Z
---

# Disable 64x128x128 GEMM config for SM86 and 89

候选状态：**defer / 未审核**。架构标签是发现线索，不是设备支持证明。

- [GitHub](https://github.com/SHI-Labs/NATTEN/pull/114)
- [原始正文与讨论](../../artifacts/github/SHI-Labs--NATTEN/pr-114/20260917T164959670620Z/page.md)
- [来源清单](../../artifacts/github/SHI-Labs--NATTEN/pr-114/20260917T164959670620Z/PROVENANCE.yaml)
- [完整 PR diff](../../artifacts/github/SHI-Labs--NATTEN/pr-114/20260917T164959670620Z/diff.patch)

Head SHA: `12ed5ea7ed92c64ef1327a439382525f4e35b034` · merged

## 关键实现文件

- [src/natten/autotuner.py](../../artifacts/github/SHI-Labs--NATTEN/pr-114/20260917T164959670620Z/key-files/head/src/natten/autotuner.py)（modified，`12ed5ea7ed92`）
- [src/natten/functional.py](../../artifacts/github/SHI-Labs--NATTEN/pr-114/20260917T164959670620Z/key-files/head/src/natten/functional.py)（modified，`12ed5ea7ed92`）
