---
id: pr-natten-111
title: Fused neighborhood attention
type: pr
url: https://github.com/SHI-Labs/NATTEN/pull/111
repo: SHI-Labs/NATTEN
number: 111
architectures:
- sm89
architecture_evidence:
  sm89: 'PR 陈述 SM80 及以后走 Tensor Core；kernel_forward.h 使用 CUTLASS threadblock/warp
    MMA。对 SM89 的可迁移性需结合 #114 的资源约束判断。'
tags:
- fna
- neighborhood-attention
- cutlass-2x
- mask
kernel_types:
- attention
symptoms:
- local-attention
- boundary-semantics
status: merged
confidence: inferred
retrieved_at: '2026-09-18'
updated_at: '2024-03-10T18:57:14Z'
inclusion_reason: FNA forward：1D/2D/3D 邻域 mask、dilation、相对位置偏置与 QK→softmax→PV 融合。
local_validation: not-run
head_sha: ed95abfec5f1698f709d099c97baf113fbd6b2da
merged_at: '2024-03-08T05:35:03Z'
merge_commit_sha: 9b99173c9fe0a1370fcbb870ddafe7930f96731b
changed_files: 254
code_urls:
- https://github.com/SHI-Labs/NATTEN/blob/ed95abfec5f1698f709d099c97baf113fbd6b2da/csrc/include/natten/cuda/fna/kernel_forward.h
- https://github.com/SHI-Labs/NATTEN/blob/ed95abfec5f1698f709d099c97baf113fbd6b2da/csrc/include/natten/cuda/fna/na_utils.cuh
- https://github.com/SHI-Labs/NATTEN/blob/ed95abfec5f1698f709d099c97baf113fbd6b2da/csrc/include/natten/cuda/fna/fna_forward.cuh
evidence_api: https://api.github.com/repos/SHI-Labs/NATTEN/pulls/111
research_focus: sm89-local-attention
candidate_role: attention-mainline
priority: P0
---

# Fused neighborhood attention

[上游 PR](https://github.com/SHI-Labs/NATTEN/pull/111) · merged `2024-03-08T05:35:03Z` · 核查 2026-09-18

## 对本任务的价值

**P0 / attention-mainline**。FNA forward：1D/2D/3D 邻域 mask、dilation、相对位置偏置与 QK→softmax→PV 融合。

PR 陈述 SM80 及以后走 Tensor Core；kernel_forward.h 使用 CUTLASS threadblock/warp MMA。对 SM89 的可迁移性需结合 #114 的资源约束判断。

## 适用边界

该实现以 FP16/BF16/FP32 为入口，没有提供本任务要求的 SM89 FP8 attention。邻域边界不是本地 window4 reflect。上游含自定义 MMA/iterator 和其它架构 SIMT 分支；本地 csrc/natten 禁止复制这些内部实现，优先直接复用 vendored CUTLASS example 41。

收录代表值得研究，不代表采用。GPU parity/性能：`not-run`。

## 固定版本实现

Head：`ed95abfec5f1698f709d099c97baf113fbd6b2da`。PR 共 254 个变更文件；已抓取 202 个关键源码文件。

- [csrc/include/natten/cuda/fna/kernel_forward.h](../../artifacts/github/SHI-Labs--NATTEN/pr-111/20260917T171022986630Z/key-files/head/csrc/include/natten/cuda/fna/kernel_forward.h) · [固定 SHA 上游](https://github.com/SHI-Labs/NATTEN/blob/ed95abfec5f1698f709d099c97baf113fbd6b2da/csrc/include/natten/cuda/fna/kernel_forward.h)
- [csrc/include/natten/cuda/fna/na_utils.cuh](../../artifacts/github/SHI-Labs--NATTEN/pr-111/20260917T171022986630Z/key-files/head/csrc/include/natten/cuda/fna/na_utils.cuh) · [固定 SHA 上游](https://github.com/SHI-Labs/NATTEN/blob/ed95abfec5f1698f709d099c97baf113fbd6b2da/csrc/include/natten/cuda/fna/na_utils.cuh)
- [csrc/include/natten/cuda/fna/fna_forward.cuh](../../artifacts/github/SHI-Labs--NATTEN/pr-111/20260917T171022986630Z/key-files/head/csrc/include/natten/cuda/fna/fna_forward.cuh) · [固定 SHA 上游](https://github.com/SHI-Labs/NATTEN/blob/ed95abfec5f1698f709d099c97baf113fbd6b2da/csrc/include/natten/cuda/fna/fna_forward.cuh)

[正文、讨论、完整 diff、来源清单和其余源码](../../wiki/candidates/pr-SHI-Labs--NATTEN-111.md)。自动 Wiki 保持未审核标记，人工筛选结论以本页为准。
