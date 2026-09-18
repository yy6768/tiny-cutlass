---
id: issue-cutlass-1181
title: '[QST] What is the usage of Sm86?'
type: issue
url: https://github.com/NVIDIA/cutlass/issues/1181
repo: NVIDIA/cutlass
number: 1181
architectures:
- sm80
- sm86
- sm89
architecture_evidence:
  sm80: 维护者解释 ArchTag 表示 kernel 的最低需求；某些 SM86/89 kernel 应保留 Sm80 policy，再选择实际编译目标。 讨论对应特定模板和旧版本；FP8 等架构新增指令仍需自己的
    atom/policy。不能全局把 Sm89 替换成 Sm80。
  sm86: 维护者解释 ArchTag 表示 kernel 的最低需求；某些 SM86/89 kernel 应保留 Sm80 policy，再选择实际编译目标。 讨论对应特定模板和旧版本；FP8 等架构新增指令仍需自己的
    atom/policy。不能全局把 Sm89 替换成 Sm80。
  sm89: 维护者解释 ArchTag 表示 kernel 的最低需求；某些 SM86/89 kernel 应保留 Sm80 policy，再选择实际编译目标。 讨论对应特定模板和旧版本；FP8 等架构新增指令仍需自己的
    atom/policy。不能全局把 Sm89 替换成 Sm80。
tags:
- dispatch
- arch-tag
kernel_types:
- gemm
symptoms:
- compile-error
status: closed
confidence: source-reported
retrieved_at: '2026-09-17'
updated_at: '2023-11-11T02:22:55Z'
inclusion_reason: 维护者解释 ArchTag 表示 kernel 的最低需求；某些 SM86/89 kernel 应保留 Sm80 policy，再选择实际编译目标。
local_validation: not-run
---

# [QST] What is the usage of Sm86?

[上游 ISSUE](https://github.com/NVIDIA/cutlass/issues/1181) · 核查于 2026-09-17 · **closed**

## 实现价值

维护者解释 ArchTag 表示 kernel 的最低需求；某些 SM86/89 kernel 应保留 Sm80 policy，再选择实际编译目标。

## 使用边界

讨论对应特定模板和旧版本；FP8 等架构新增指令仍需自己的 atom/policy。不能全局把 Sm89 替换成 Sm80。

本地 GPU 验证：未运行。

## 讨论入口

- [讨论 1806632858](https://github.com/NVIDIA/cutlass/issues/1181#issuecomment-1806632858)（2023-11-11）
- [讨论 1806634052](https://github.com/NVIDIA/cutlass/issues/1181#issuecomment-1806634052)（2023-11-11）
