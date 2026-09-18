---
id: issue-cutlass-3044
title: 'CuTe DSL: FP8 MMA segfaults on SM120 — MmaAtomSM80Type missing kind::f8f6f4 lowering'
type: issue
url: https://github.com/NVIDIA/cutlass/issues/3044
repo: NVIDIA/cutlass
number: 3044
architectures:
- sm120
architecture_evidence:
  sm120: 记录把 SM80/SM89 FP8 atom 用于 SM120 lowering 时的编译崩溃与不同指令格式。 历史版本问题，closed 不证明所有 CuTe DSL 版本均修好；复现还涉及 SM121。
tags:
- fp8
- cute-dsl
- mma-sync
kernel_types:
- gemm
- attention
symptoms:
- compiler-crash
status: closed
confidence: source-reported
retrieved_at: '2026-09-17'
updated_at: '2026-07-04T09:40:23Z'
inclusion_reason: 记录把 SM80/SM89 FP8 atom 用于 SM120 lowering 时的编译崩溃与不同指令格式。
local_validation: not-run
---

# CuTe DSL: FP8 MMA segfaults on SM120 — MmaAtomSM80Type missing kind::f8f6f4 lowering

[上游 ISSUE](https://github.com/NVIDIA/cutlass/issues/3044) · 核查于 2026-09-17 · **closed**

## 实现价值

记录把 SM80/SM89 FP8 atom 用于 SM120 lowering 时的编译崩溃与不同指令格式。

## 使用边界

历史版本问题，closed 不证明所有 CuTe DSL 版本均修好；复现还涉及 SM121。

本地 GPU 验证：未运行。

## 讨论入口

- [讨论 4121490419](https://github.com/NVIDIA/cutlass/issues/3044#issuecomment-4121490419)（2026-03-24）
- [讨论 4320722481](https://github.com/NVIDIA/cutlass/issues/3044#issuecomment-4320722481)（2026-04-25）
- [讨论 4862669254](https://github.com/NVIDIA/cutlass/issues/3044#issuecomment-4862669254)（2026-07-02）
