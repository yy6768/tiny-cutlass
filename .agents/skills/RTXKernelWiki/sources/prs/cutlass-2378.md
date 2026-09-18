---
id: pr-cutlass-2378
title: support fp16 accmulator for sm89 fp8 mma
type: pr
url: https://github.com/NVIDIA/cutlass/pull/2378
repo: NVIDIA/cutlass
number: 2378
architectures:
- sm89
architecture_evidence:
  sm89: 补充 SM89 FP8 MMA 的 FP16 accumulator 路径及测试。 FP16 累加改变误差与溢出行为；不能按 FP32 accumulation 的容差或精度结论使用。
tags:
- fp8
- mma-sync
- accumulation
kernel_types:
- gemm
symptoms:
- numerical-accuracy
status: merged
confidence: source-reported
retrieved_at: '2026-09-17'
updated_at: '2025-07-31T02:12:09Z'
inclusion_reason: 补充 SM89 FP8 MMA 的 FP16 accumulator 路径及测试。
local_validation: not-run
head_sha: 954077ba9fa0e431e120b11e5e0c46b93c82cabf
merged_at: '2025-07-31T02:12:09Z'
merge_commit_sha: 26b74500232f254779f5b5d70c5ed98fb6366612
changed_files: 3
code_urls:
- https://github.com/kf-zhang/cutlass/blob/954077ba9fa0e431e120b11e5e0c46b93c82cabf/include/cute/arch/mma_sm89.hpp
- https://github.com/kf-zhang/cutlass/blob/954077ba9fa0e431e120b11e5e0c46b93c82cabf/include/cute/atom/mma_traits_sm89.hpp
- https://github.com/kf-zhang/cutlass/blob/954077ba9fa0e431e120b11e5e0c46b93c82cabf/test/unit/cute/ampere/cooperative_gemm.cu
evidence_api: https://api.github.com/repos/NVIDIA/cutlass/pulls/2378
---

# support fp16 accmulator for sm89 fp8 mma

[上游 PR](https://github.com/NVIDIA/cutlass/pull/2378) · 核查于 2026-09-17 · **merged**

## 实现价值

补充 SM89 FP8 MMA 的 FP16 accumulator 路径及测试。

## 使用边界

FP16 累加改变误差与溢出行为；不能按 FP32 accumulation 的容差或精度结论使用。

本地 GPU 验证：未运行。

## 固定版本源码

PR head：`954077ba9fa0e431e120b11e5e0c46b93c82cabf`。共 3 个 changed files；以下是实现阅读入口，不是完整变更清单。

- [include/cute/arch/mma_sm89.hpp](https://github.com/kf-zhang/cutlass/blob/954077ba9fa0e431e120b11e5e0c46b93c82cabf/include/cute/arch/mma_sm89.hpp)
- [include/cute/atom/mma_traits_sm89.hpp](https://github.com/kf-zhang/cutlass/blob/954077ba9fa0e431e120b11e5e0c46b93c82cabf/include/cute/atom/mma_traits_sm89.hpp)
- [test/unit/cute/ampere/cooperative_gemm.cu](https://github.com/kf-zhang/cutlass/blob/954077ba9fa0e431e120b11e5e0c46b93c82cabf/test/unit/cute/ampere/cooperative_gemm.cu)

## 讨论入口

- [讨论 3055968348](https://github.com/NVIDIA/cutlass/pull/2378#issuecomment-3055968348)（2025-07-10）
- [讨论 3111742181](https://github.com/NVIDIA/cutlass/pull/2378#issuecomment-3111742181)（2025-07-24）
- [讨论 3135192206](https://github.com/NVIDIA/cutlass/pull/2378#issuecomment-3135192206)（2025-07-30）
