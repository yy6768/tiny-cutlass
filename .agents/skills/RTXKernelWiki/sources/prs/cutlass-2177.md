---
id: pr-cutlass-2177
title: add support for sm89 in cute and the unit tests
type: pr
url: https://github.com/NVIDIA/cutlass/pull/2177
repo: NVIDIA/cutlass
number: 2177
architectures:
- sm89
architecture_evidence:
  sm89: '引入 Ada CuTe FP8 MMA atom、traits 和对应单元测试。 检查具体 accumulator specialization；FP16 累加的后续补充见 #2378。'
tags:
- fp8
- mma-sync
- cute
kernel_types:
- gemm
symptoms:
- implementation-choice
status: merged
confidence: source-reported
retrieved_at: '2026-09-17'
updated_at: '2025-05-04T06:01:17Z'
inclusion_reason: 引入 Ada CuTe FP8 MMA atom、traits 和对应单元测试。
local_validation: not-run
head_sha: 152947dbbd75921e0914dcd851319546b84b3d12
merged_at: '2025-04-10T18:16:36Z'
merge_commit_sha: 19cc2a5febed01ae76667ffcd3c073c38c47065b
changed_files: 4
code_urls:
- https://github.com/kf-zhang/cutlass/blob/152947dbbd75921e0914dcd851319546b84b3d12/include/cute/arch/mma_sm89.hpp
- https://github.com/kf-zhang/cutlass/blob/152947dbbd75921e0914dcd851319546b84b3d12/include/cute/atom/mma_atom.hpp
- https://github.com/kf-zhang/cutlass/blob/152947dbbd75921e0914dcd851319546b84b3d12/include/cute/atom/mma_traits_sm89.hpp
- https://github.com/kf-zhang/cutlass/blob/152947dbbd75921e0914dcd851319546b84b3d12/test/unit/cute/ampere/cooperative_gemm.cu
evidence_api: https://api.github.com/repos/NVIDIA/cutlass/pulls/2177
---

# add support for sm89 in cute and the unit tests

[上游 PR](https://github.com/NVIDIA/cutlass/pull/2177) · 核查于 2026-09-17 · **merged**

## 实现价值

引入 Ada CuTe FP8 MMA atom、traits 和对应单元测试。

## 使用边界

检查具体 accumulator specialization；FP16 累加的后续补充见 #2378。

本地 GPU 验证：未运行。

## 固定版本源码

PR head：`152947dbbd75921e0914dcd851319546b84b3d12`。共 4 个 changed files；以下是实现阅读入口，不是完整变更清单。

- [include/cute/arch/mma_sm89.hpp](https://github.com/kf-zhang/cutlass/blob/152947dbbd75921e0914dcd851319546b84b3d12/include/cute/arch/mma_sm89.hpp)
- [include/cute/atom/mma_atom.hpp](https://github.com/kf-zhang/cutlass/blob/152947dbbd75921e0914dcd851319546b84b3d12/include/cute/atom/mma_atom.hpp)
- [include/cute/atom/mma_traits_sm89.hpp](https://github.com/kf-zhang/cutlass/blob/152947dbbd75921e0914dcd851319546b84b3d12/include/cute/atom/mma_traits_sm89.hpp)
- [test/unit/cute/ampere/cooperative_gemm.cu](https://github.com/kf-zhang/cutlass/blob/152947dbbd75921e0914dcd851319546b84b3d12/test/unit/cute/ampere/cooperative_gemm.cu)

## 讨论入口

- [讨论 2765995035](https://github.com/NVIDIA/cutlass/pull/2177#issuecomment-2765995035)（2025-03-31）
- [讨论 2848393155](https://github.com/NVIDIA/cutlass/pull/2177#issuecomment-2848393155)（2025-05-03）
- [讨论 2849027055](https://github.com/NVIDIA/cutlass/pull/2177#issuecomment-2849027055)（2025-05-04）
