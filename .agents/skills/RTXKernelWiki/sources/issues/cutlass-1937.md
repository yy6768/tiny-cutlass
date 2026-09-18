---
id: issue-cutlass-1937
title: '[QST] FP8 with row-wise scaling on Ada-Lovelace'
type: issue
url: https://github.com/NVIDIA/cutlass/issues/1937
repo: NVIDIA/cutlass
number: 1937
architectures:
- sm89
architecture_evidence:
  sm89: Ada FP8 row-wise scaling 讨论提供 EVT 生成代码的思路。 旧 issue 的缺功能陈述不等于当前版本仍缺失；检查生成 C++、缩放维度与运行时目标。
tags:
- fp8
- scaling
- evt
kernel_types:
- gemm
symptoms:
- implementation-choice
status: open
confidence: source-reported
retrieved_at: '2026-09-17'
updated_at: '2025-03-14T23:05:57Z'
inclusion_reason: Ada FP8 row-wise scaling 讨论提供 EVT 生成代码的思路。
local_validation: not-run
---

# [QST] FP8 with row-wise scaling on Ada-Lovelace

[上游 ISSUE](https://github.com/NVIDIA/cutlass/issues/1937) · 核查于 2026-09-17 · **open**

## 实现价值

Ada FP8 row-wise scaling 讨论提供 EVT 生成代码的思路。

## 使用边界

旧 issue 的缺功能陈述不等于当前版本仍缺失；检查生成 C++、缩放维度与运行时目标。

本地 GPU 验证：未运行。

## 讨论入口

- [讨论 2474193538](https://github.com/NVIDIA/cutlass/issues/1937#issuecomment-2474193538)（2024-11-13）
- [讨论 2474410335](https://github.com/NVIDIA/cutlass/issues/1937#issuecomment-2474410335)（2024-11-13）
- [讨论 2477546629](https://github.com/NVIDIA/cutlass/issues/1937#issuecomment-2477546629)（2024-11-14）
