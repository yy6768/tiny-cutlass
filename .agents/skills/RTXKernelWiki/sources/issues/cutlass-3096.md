---
id: issue-cutlass-3096
title: 'SM120 (Bug) (With FIx)(RTX Blackwell) NVFP4 MoE: CUTLASS Grouped GEMM Produces Garbage Output; Fixed via
  FlashInfer SM120 Patches + compute_120f (CUDA 13.0) — 39 tok/s Native FP4'
type: issue
url: https://github.com/NVIDIA/cutlass/issues/3096
repo: NVIDIA/cutlass
number: 3096
architectures:
- sm120
architecture_evidence:
  sm120: 记录 RTX PRO 6000 上 NVFP4 grouped GEMM 的错误输出、初始化失败与工具链排查。 正文与后续根因讨论有演变；按具体 commit/shape 复现，不整体采纳标题中的 fixed
    或端到端吞吐。
tags:
- nvfp4
- grouped-gemm
- tma
kernel_types:
- grouped-gemm
- moe
symptoms:
- correctness
status: closed
confidence: source-reported
retrieved_at: '2026-09-17'
updated_at: '2026-05-16T02:55:27Z'
inclusion_reason: 记录 RTX PRO 6000 上 NVFP4 grouped GEMM 的错误输出、初始化失败与工具链排查。
local_validation: not-run
---

# SM120 (Bug) (With FIx)(RTX Blackwell) NVFP4 MoE: CUTLASS Grouped GEMM Produces Garbage Output; Fixed via FlashInfer SM120 Patches + compute_120f (CUDA 13.0) — 39 tok/s Native FP4

[上游 ISSUE](https://github.com/NVIDIA/cutlass/issues/3096) · 核查于 2026-09-17 · **closed**

## 实现价值

记录 RTX PRO 6000 上 NVFP4 grouped GEMM 的错误输出、初始化失败与工具链排查。

## 使用边界

正文与后续根因讨论有演变；按具体 commit/shape 复现，不整体采纳标题中的 fixed 或端到端吞吐。

本地 GPU 验证：未运行。

## 讨论入口

- [讨论 4056827914](https://github.com/NVIDIA/cutlass/issues/3096#issuecomment-4056827914)（2026-03-13）
- [讨论 4465204642](https://github.com/NVIDIA/cutlass/issues/3096#issuecomment-4465204642)（2026-05-16）
- [讨论 4465257666](https://github.com/NVIDIA/cutlass/issues/3096#issuecomment-4465257666)（2026-05-16）
