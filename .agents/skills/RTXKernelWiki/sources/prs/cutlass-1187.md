---
id: pr-cutlass-1187
title: fix cp.async L2 prefetch typo
type: pr
url: https://github.com/NVIDIA/cutlass/pull/1187
repo: NVIDIA/cutlass
number: 1187
architectures:
- sm80
- sm86
- sm89
architecture_evidence:
  sm80: 修正 CuTe SM80 cp.async 的 L2 prefetch 拼写；从 copy atom 跟踪真正发出的 PTX。 SM80 是直接源码证据；SM86/89 为共享指令路径的迁移线索。检查当前版本的
    zfill/predication 语义，不由旧讨论推导所有 copy 都走同一路径。
  sm86: 修正 CuTe SM80 cp.async 的 L2 prefetch 拼写；从 copy atom 跟踪真正发出的 PTX。 SM80 是直接源码证据；SM86/89 为共享指令路径的迁移线索。检查当前版本的
    zfill/predication 语义，不由旧讨论推导所有 copy 都走同一路径。
  sm89: 修正 CuTe SM80 cp.async 的 L2 prefetch 拼写；从 copy atom 跟踪真正发出的 PTX。 SM80 是直接源码证据；SM86/89 为共享指令路径的迁移线索。检查当前版本的
    zfill/predication 语义，不由旧讨论推导所有 copy 都走同一路径。
tags:
- cp-async
- pipeline
kernel_types:
- gemm
- convolution
symptoms:
- memory-stall
status: merged
confidence: inferred
retrieved_at: '2026-09-17'
updated_at: '2023-11-28T21:58:04Z'
inclusion_reason: 修正 CuTe SM80 cp.async 的 L2 prefetch 拼写；从 copy atom 跟踪真正发出的 PTX。
local_validation: not-run
head_sha: 21424b0c5fa2bd31e8fa5655ee4252b7a01953d1
merged_at: '2023-11-28T21:58:04Z'
merge_commit_sha: eb01d5449d616fb1bde45777e14fed8f7fbda509
changed_files: 1
code_urls:
- https://github.com/reed-lau/cutlass/blob/21424b0c5fa2bd31e8fa5655ee4252b7a01953d1/include/cute/arch/copy_sm80.hpp
evidence_api: https://api.github.com/repos/NVIDIA/cutlass/pulls/1187
---

# fix cp.async L2 prefetch typo

[上游 PR](https://github.com/NVIDIA/cutlass/pull/1187) · 核查于 2026-09-17 · **merged**

## 实现价值

修正 CuTe SM80 cp.async 的 L2 prefetch 拼写；从 copy atom 跟踪真正发出的 PTX。

## 使用边界

SM80 是直接源码证据；SM86/89 为共享指令路径的迁移线索。检查当前版本的 zfill/predication 语义，不由旧讨论推导所有 copy 都走同一路径。

本地 GPU 验证：未运行。

## 固定版本源码

PR head：`21424b0c5fa2bd31e8fa5655ee4252b7a01953d1`。共 1 个 changed files；以下是实现阅读入口，不是完整变更清单。

- [include/cute/arch/copy_sm80.hpp](https://github.com/reed-lau/cutlass/blob/21424b0c5fa2bd31e8fa5655ee4252b7a01953d1/include/cute/arch/copy_sm80.hpp)

## 讨论入口

- [讨论 1811180380](https://github.com/NVIDIA/cutlass/pull/1187#issuecomment-1811180380)（2023-11-14）
- [讨论 1811183930](https://github.com/NVIDIA/cutlass/pull/1187#issuecomment-1811183930)（2023-11-14）
- [讨论 1827225128](https://github.com/NVIDIA/cutlass/pull/1187#issuecomment-1827225128)（2023-11-27）
