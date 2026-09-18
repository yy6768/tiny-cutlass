---
id: pr-cutlass-3055
title: Replace std::min with cute::min in sm120 blockwise scaling device functions
type: pr
url: https://github.com/NVIDIA/cutlass/pull/3055
repo: NVIDIA/cutlass
number: 3055
architectures:
- sm120
architecture_evidence:
  sm120: 把 SM120 blockwise scaling device 函数里的 std::min 替换成可用于 device 的 cute::min。 这是具体编译修复，不证明相关 grouped GEMM 的所有
    shape 均正确。
tags:
- fp8
- block-scaling
- cuda-cpp
kernel_types:
- gemm
- grouped-gemm
symptoms:
- compile-error
status: merged
confidence: source-reported
retrieved_at: '2026-09-17'
updated_at: '2026-04-24T15:13:39Z'
inclusion_reason: 把 SM120 blockwise scaling device 函数里的 std::min 替换成可用于 device 的 cute::min。
local_validation: not-run
head_sha: c969a1b1f9bea269e9a8f809a00732d4c2465553
merged_at: '2026-04-24T15:13:38Z'
merge_commit_sha: 9135a9bb6dbbe55e6d251eaed76ca3e7a95aa32a
changed_files: 2
code_urls:
- https://github.com/vruga/cutlass/blob/c969a1b1f9bea269e9a8f809a00732d4c2465553/include/cutlass/gemm/collective/sm120_mma_array_tma_blockwise_scaling.hpp
- https://github.com/vruga/cutlass/blob/c969a1b1f9bea269e9a8f809a00732d4c2465553/include/cutlass/gemm/collective/sm120_mma_tma_blockwise_scaling.hpp
evidence_api: https://api.github.com/repos/NVIDIA/cutlass/pulls/3055
---

# Replace std::min with cute::min in sm120 blockwise scaling device functions

[上游 PR](https://github.com/NVIDIA/cutlass/pull/3055) · 核查于 2026-09-17 · **merged**

## 实现价值

把 SM120 blockwise scaling device 函数里的 std::min 替换成可用于 device 的 cute::min。

## 使用边界

这是具体编译修复，不证明相关 grouped GEMM 的所有 shape 均正确。

本地 GPU 验证：未运行。

## 固定版本源码

PR head：`c969a1b1f9bea269e9a8f809a00732d4c2465553`。共 2 个 changed files；以下是实现阅读入口，不是完整变更清单。

- [include/cutlass/gemm/collective/sm120_mma_array_tma_blockwise_scaling.hpp](https://github.com/vruga/cutlass/blob/c969a1b1f9bea269e9a8f809a00732d4c2465553/include/cutlass/gemm/collective/sm120_mma_array_tma_blockwise_scaling.hpp)
- [include/cutlass/gemm/collective/sm120_mma_tma_blockwise_scaling.hpp](https://github.com/vruga/cutlass/blob/c969a1b1f9bea269e9a8f809a00732d4c2465553/include/cutlass/gemm/collective/sm120_mma_tma_blockwise_scaling.hpp)

## 讨论入口

- [讨论 4149418002](https://github.com/NVIDIA/cutlass/pull/3055#issuecomment-4149418002)（2026-03-29）
