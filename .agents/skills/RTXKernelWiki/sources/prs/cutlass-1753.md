---
id: pr-cutlass-1753
title: Fix EVT for cutlass::gemm::kernel::DefaultGemmWithVisitor's behavior when constructing GemmUniversalAdapter
type: pr
url: https://github.com/NVIDIA/cutlass/pull/1753
repo: NVIDIA/cutlass
number: 1753
architectures:
- sm80
- sm89
architecture_evidence:
  sm80: 修复 DefaultGemmWithVisitor 配合 GemmUniversalAdapter 时的继承与类型可见性。 SM80 示例直接复现；SM89 相关性来自 CUTLASS 2.x EVT 的迁移。不能推导所有
    convolution device API 支持 EVT。
  sm89: 修复 DefaultGemmWithVisitor 配合 GemmUniversalAdapter 时的继承与类型可见性。 SM80 示例直接复现；SM89 相关性来自 CUTLASS 2.x EVT 的迁移。不能推导所有
    convolution device API 支持 EVT。
tags:
- epilogue
- evt
- fusion
kernel_types:
- gemm
symptoms:
- compile-error
status: merged
confidence: inferred
retrieved_at: '2026-09-17'
updated_at: '2024-10-23T16:45:59Z'
inclusion_reason: 修复 DefaultGemmWithVisitor 配合 GemmUniversalAdapter 时的继承与类型可见性。
local_validation: not-run
head_sha: 07de9b6f60f78c6466a894cc73d239d59ff7315f
merged_at: '2024-10-23T16:45:58Z'
merge_commit_sha: b0c09ed07762e7f3fa49b2953a5dd17e851231a6
changed_files: 1
code_urls:
- https://github.com/Xinyu302/cutlass/blob/07de9b6f60f78c6466a894cc73d239d59ff7315f/include/cutlass/gemm/kernel/gemm_universal_with_visitor.h
evidence_api: https://api.github.com/repos/NVIDIA/cutlass/pulls/1753
---

# Fix EVT for cutlass::gemm::kernel::DefaultGemmWithVisitor's behavior when constructing GemmUniversalAdapter

[上游 PR](https://github.com/NVIDIA/cutlass/pull/1753) · 核查于 2026-09-17 · **merged**

## 实现价值

修复 DefaultGemmWithVisitor 配合 GemmUniversalAdapter 时的继承与类型可见性。

## 使用边界

SM80 示例直接复现；SM89 相关性来自 CUTLASS 2.x EVT 的迁移。不能推导所有 convolution device API 支持 EVT。

本地 GPU 验证：未运行。

## 固定版本源码

PR head：`07de9b6f60f78c6466a894cc73d239d59ff7315f`。共 1 个 changed files；以下是实现阅读入口，不是完整变更清单。

- [include/cutlass/gemm/kernel/gemm_universal_with_visitor.h](https://github.com/Xinyu302/cutlass/blob/07de9b6f60f78c6466a894cc73d239d59ff7315f/include/cutlass/gemm/kernel/gemm_universal_with_visitor.h)

## 讨论入口

- [讨论 2345425551](https://github.com/NVIDIA/cutlass/pull/1753#issuecomment-2345425551)（2024-09-12）
- [讨论 2362536819](https://github.com/NVIDIA/cutlass/pull/1753#issuecomment-2362536819)（2024-09-20）
- [讨论 2366855204](https://github.com/NVIDIA/cutlass/pull/1753#issuecomment-2366855204)（2024-09-22）
