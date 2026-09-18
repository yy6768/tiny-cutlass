---
id: pr-cutlass-3196
title: Small Tile M BlockScaled GEMM + Grouped GEMM on SM12x
type: pr
url: https://github.com/NVIDIA/cutlass/pull/3196
repo: NVIDIA/cutlass
number: 3196
architectures:
- sm120
architecture_evidence:
  sm120: 提出 SM12x blockscaled 小 M/K tile、scale padding 与 grouped GEMM 扩展。 closed-unmerged 候选；包含 SM121 信息，不能把 GB10
    测试当成 RTX 实测。
tags:
- nvfp4
- tiling
- grouped-gemm
kernel_types:
- gemm
- grouped-gemm
- attention
symptoms:
- resource-limit
status: closed-unmerged
confidence: source-reported
retrieved_at: '2026-09-17'
updated_at: '2026-05-06T01:38:55Z'
inclusion_reason: 提出 SM12x blockscaled 小 M/K tile、scale padding 与 grouped GEMM 扩展。
local_validation: not-run
head_sha: a27f9c7a33e02c60e05ddfeea3f9baf24138ef60
merged_at: null
merge_commit_sha: 673a034843dcf042edb2a3c0de6fdbffca722197
changed_files: 9
code_urls:
- https://github.com/thedatamates/cutlass/blob/a27f9c7a33e02c60e05ddfeea3f9baf24138ef60/include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl
- https://github.com/thedatamates/cutlass/blob/a27f9c7a33e02c60e05ddfeea3f9baf24138ef60/include/cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp
- https://github.com/thedatamates/cutlass/blob/a27f9c7a33e02c60e05ddfeea3f9baf24138ef60/include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp
- https://github.com/thedatamates/cutlass/blob/a27f9c7a33e02c60e05ddfeea3f9baf24138ef60/include/cutlass/gemm/kernel/sm90_gemm_array_tma_warpspecialized_cooperative.hpp
- https://github.com/thedatamates/cutlass/blob/a27f9c7a33e02c60e05ddfeea3f9baf24138ef60/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp
- https://github.com/thedatamates/cutlass/blob/a27f9c7a33e02c60e05ddfeea3f9baf24138ef60/python/cutlass_library/generator.py
evidence_api: https://api.github.com/repos/NVIDIA/cutlass/pulls/3196
---

# Small Tile M BlockScaled GEMM + Grouped GEMM on SM12x

[上游 PR](https://github.com/NVIDIA/cutlass/pull/3196) · 核查于 2026-09-17 · **closed-unmerged**

## 实现价值

提出 SM12x blockscaled 小 M/K tile、scale padding 与 grouped GEMM 扩展。

## 使用边界

closed-unmerged 候选；包含 SM121 信息，不能把 GB10 测试当成 RTX 实测。

本地 GPU 验证：未运行。

## 固定版本源码

PR head：`a27f9c7a33e02c60e05ddfeea3f9baf24138ef60`。共 9 个 changed files；以下是实现阅读入口，不是完整变更清单。

- [include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl](https://github.com/thedatamates/cutlass/blob/a27f9c7a33e02c60e05ddfeea3f9baf24138ef60/include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl)
- [include/cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp](https://github.com/thedatamates/cutlass/blob/a27f9c7a33e02c60e05ddfeea3f9baf24138ef60/include/cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp)
- [include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp](https://github.com/thedatamates/cutlass/blob/a27f9c7a33e02c60e05ddfeea3f9baf24138ef60/include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp)
- [include/cutlass/gemm/kernel/sm90_gemm_array_tma_warpspecialized_cooperative.hpp](https://github.com/thedatamates/cutlass/blob/a27f9c7a33e02c60e05ddfeea3f9baf24138ef60/include/cutlass/gemm/kernel/sm90_gemm_array_tma_warpspecialized_cooperative.hpp)
- [include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp](https://github.com/thedatamates/cutlass/blob/a27f9c7a33e02c60e05ddfeea3f9baf24138ef60/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp)
- [python/cutlass_library/generator.py](https://github.com/thedatamates/cutlass/blob/a27f9c7a33e02c60e05ddfeea3f9baf24138ef60/python/cutlass_library/generator.py)

## 讨论入口

- [讨论 4354532812](https://github.com/NVIDIA/cutlass/pull/3196#issuecomment-4354532812)（2026-04-30）
- [讨论 4364096443](https://github.com/NVIDIA/cutlass/pull/3196#issuecomment-4364096443)（2026-05-02）
