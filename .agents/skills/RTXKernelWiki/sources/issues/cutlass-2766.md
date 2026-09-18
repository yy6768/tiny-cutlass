---
id: issue-cutlass-2766
title: '[QST] Does sm120 gemm kernels support fp16/tf32 inputs?'
type: issue
url: https://github.com/NVIDIA/cutlass/issues/2766
repo: NVIDIA/cutlass
number: 2766
architectures:
- sm120
architecture_evidence:
  sm120: 解释 SM100 GEMM 不能直接用于 SM120，并讨论 FP16/TF32 kernel 选择。 结合当前 PTX 与官方示例判断；生成器暂未产出某种 kernel 不代表硬件不支持该 dtype。
tags:
- mma-sync
- dispatch
kernel_types:
- gemm
symptoms:
- unsupported-instruction
status: closed
confidence: source-reported
retrieved_at: '2026-09-17'
updated_at: '2026-07-22T17:05:04Z'
inclusion_reason: 解释 SM100 GEMM 不能直接用于 SM120，并讨论 FP16/TF32 kernel 选择。
local_validation: not-run
---

# [QST] Does sm120 gemm kernels support fp16/tf32 inputs?

[上游 ISSUE](https://github.com/NVIDIA/cutlass/issues/2766) · 核查于 2026-09-17 · **closed**

## 实现价值

解释 SM100 GEMM 不能直接用于 SM120，并讨论 FP16/TF32 kernel 选择。

## 使用边界

结合当前 PTX 与官方示例判断；生成器暂未产出某种 kernel 不代表硬件不支持该 dtype。

本地 GPU 验证：未运行。

## 讨论入口

- [讨论 3540219211](https://github.com/NVIDIA/cutlass/issues/2766#issuecomment-3540219211)（2025-11-17）
- [讨论 3666192305](https://github.com/NVIDIA/cutlass/issues/2766#issuecomment-3666192305)（2025-12-17）
- [讨论 3668050024](https://github.com/NVIDIA/cutlass/issues/2766#issuecomment-3668050024)（2025-12-18）
