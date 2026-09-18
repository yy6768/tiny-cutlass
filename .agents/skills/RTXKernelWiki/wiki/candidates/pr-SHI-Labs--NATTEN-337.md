---
id: candidate-pr-SHI-Labs--NATTEN-337
title: 'cutlass-fna: 64b strides'
type: pr
repo: SHI-Labs/NATTEN
architectures:
- sm89
status: merged
tags: []
review: defer
url: https://github.com/SHI-Labs/NATTEN/pull/337
inclusion_reason: 自动抓取 candidate；架构来自搜索命中，需阅读代码确认。
local_validation: not-run
retrieved_at: 20260917T164959673620Z
artifact_dir: artifacts/github/SHI-Labs--NATTEN/pr-337/20260917T164959673620Z
---

# cutlass-fna: 64b strides

候选状态：**defer / 未审核**。架构标签是发现线索，不是设备支持证明。

- [GitHub](https://github.com/SHI-Labs/NATTEN/pull/337)
- [原始正文与讨论](../../artifacts/github/SHI-Labs--NATTEN/pr-337/20260917T164959673620Z/page.md)
- [来源清单](../../artifacts/github/SHI-Labs--NATTEN/pr-337/20260917T164959673620Z/PROVENANCE.yaml)
- [完整 PR diff](../../artifacts/github/SHI-Labs--NATTEN/pr-337/20260917T164959673620Z/diff.patch)

Head SHA: `6fd34669017a39cd37b10d9edd3c65c88c18dfb6` · merged

## 关键实现文件

- [csrc/include/natten/cuda/fna/epilogue/predicated_tile_iterator.h](../../artifacts/github/SHI-Labs--NATTEN/pr-337/20260917T164959673620Z/key-files/head/csrc/include/natten/cuda/fna/epilogue/predicated_tile_iterator.h)（modified，`6fd34669017a`）
- [csrc/include/natten/cuda/fna/epilogue/predicated_tile_iterator_params.h](../../artifacts/github/SHI-Labs--NATTEN/pr-337/20260917T164959673620Z/key-files/head/csrc/include/natten/cuda/fna/epilogue/predicated_tile_iterator_params.h)（modified，`6fd34669017a`）
- [csrc/include/natten/cuda/fna/iterators/epilogue_predicated_tile_iterator.h](../../artifacts/github/SHI-Labs--NATTEN/pr-337/20260917T164959673620Z/key-files/head/csrc/include/natten/cuda/fna/iterators/epilogue_predicated_tile_iterator.h)（modified，`6fd34669017a`）
- [csrc/include/natten/cuda/fna/iterators/predicated_tile_access_iterator.h](../../artifacts/github/SHI-Labs--NATTEN/pr-337/20260917T164959673620Z/key-files/head/csrc/include/natten/cuda/fna/iterators/predicated_tile_access_iterator.h)（modified，`6fd34669017a`）
- [csrc/include/natten/cuda/fna/iterators/predicated_tile_access_iterator_residual_last.h](../../artifacts/github/SHI-Labs--NATTEN/pr-337/20260917T164959673620Z/key-files/head/csrc/include/natten/cuda/fna/iterators/predicated_tile_access_iterator_residual_last.h)（modified，`6fd34669017a`）
- [csrc/include/natten/cuda/fna/iterators/predicated_tile_iterator.h](../../artifacts/github/SHI-Labs--NATTEN/pr-337/20260917T164959673620Z/key-files/head/csrc/include/natten/cuda/fna/iterators/predicated_tile_iterator.h)（modified，`6fd34669017a`）
- [csrc/include/natten/cuda/fna/iterators/predicated_tile_iterator_residual_last.h](../../artifacts/github/SHI-Labs--NATTEN/pr-337/20260917T164959673620Z/key-files/head/csrc/include/natten/cuda/fna/iterators/predicated_tile_iterator_residual_last.h)（modified，`6fd34669017a`）
- [csrc/include/natten/cuda/fna/kernel_backward.h](../../artifacts/github/SHI-Labs--NATTEN/pr-337/20260917T164959673620Z/key-files/head/csrc/include/natten/cuda/fna/kernel_backward.h)（modified，`6fd34669017a`）
- [csrc/include/natten/cuda/fna/kernel_forward.h](../../artifacts/github/SHI-Labs--NATTEN/pr-337/20260917T164959673620Z/key-files/head/csrc/include/natten/cuda/fna/kernel_forward.h)（modified，`6fd34669017a`）
- [csrc/include/natten/cuda/fna/na_utils.cuh](../../artifacts/github/SHI-Labs--NATTEN/pr-337/20260917T164959673620Z/key-files/head/csrc/include/natten/cuda/fna/na_utils.cuh)（modified，`6fd34669017a`）
