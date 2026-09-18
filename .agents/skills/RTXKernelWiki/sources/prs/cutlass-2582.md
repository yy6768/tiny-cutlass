---
id: pr-cutlass-2582
title: Fix Copy_Atom type mismatch in sgemm_sm80.cu
type: pr
url: https://github.com/NVIDIA/cutlass/pull/2582
repo: NVIDIA/cutlass
number: 2582
architectures:
- sm80
- sm86
- sm89
architecture_evidence:
  sm80: 修正 sgemm_sm80 泛型 gemm_nt/gemm_tn 的 Copy_Atom 类型不匹配。 默认 half specialization 能编译不代表泛型入口正确；SM86/89 需按自己的 dtype/layout
    实例化验证。
  sm86: 修正 sgemm_sm80 泛型 gemm_nt/gemm_tn 的 Copy_Atom 类型不匹配。 默认 half specialization 能编译不代表泛型入口正确；SM86/89 需按自己的 dtype/layout
    实例化验证。
  sm89: 修正 sgemm_sm80 泛型 gemm_nt/gemm_tn 的 Copy_Atom 类型不匹配。 默认 half specialization 能编译不代表泛型入口正确；SM86/89 需按自己的 dtype/layout
    实例化验证。
tags:
- cute
- copy-atom
kernel_types:
- gemm
symptoms:
- compile-error
status: merged
confidence: inferred
retrieved_at: '2026-09-17'
updated_at: '2025-09-04T23:56:18Z'
inclusion_reason: 修正 sgemm_sm80 泛型 gemm_nt/gemm_tn 的 Copy_Atom 类型不匹配。
local_validation: not-run
head_sha: d7df8bf1cbb33d58d8048116509255ce9a345381
merged_at: '2025-09-04T23:56:17Z'
merge_commit_sha: b6ccf34aef08a56195ce8465807d35a51e01acae
changed_files: 1
code_urls:
- https://github.com/lifuhuang/cutlass/blob/d7df8bf1cbb33d58d8048116509255ce9a345381/examples/cute/tutorial/sgemm_sm80.cu
evidence_api: https://api.github.com/repos/NVIDIA/cutlass/pulls/2582
---

# Fix Copy_Atom type mismatch in sgemm_sm80.cu

[上游 PR](https://github.com/NVIDIA/cutlass/pull/2582) · 核查于 2026-09-17 · **merged**

## 实现价值

修正 sgemm_sm80 泛型 gemm_nt/gemm_tn 的 Copy_Atom 类型不匹配。

## 使用边界

默认 half specialization 能编译不代表泛型入口正确；SM86/89 需按自己的 dtype/layout 实例化验证。

本地 GPU 验证：未运行。

## 固定版本源码

PR head：`d7df8bf1cbb33d58d8048116509255ce9a345381`。共 1 个 changed files；以下是实现阅读入口，不是完整变更清单。

- [examples/cute/tutorial/sgemm_sm80.cu](https://github.com/lifuhuang/cutlass/blob/d7df8bf1cbb33d58d8048116509255ce9a345381/examples/cute/tutorial/sgemm_sm80.cu)

## 讨论入口

- [讨论 3252420397](https://github.com/NVIDIA/cutlass/pull/2582#issuecomment-3252420397)（2025-09-04）
