---
id: issue-cutlass-2053
title: '[QST] Grouped GEMM on A10 GPUs'
type: issue
url: https://github.com/NVIDIA/cutlass/issues/2053
repo: NVIDIA/cutlass
number: 2053
architectures:
- sm80
- sm86
architecture_evidence:
  sm80: A100 上工作的 grouped GEMM 在 A10 上失败；讨论通过减小 ThreadblockShape 处理 SMEM 预算。 SM80/86 直接出现于设备与模板；tile/stages 必须根据目标设备资源重新选。
  sm86: A100 上工作的 grouped GEMM 在 A10 上失败；讨论通过减小 ThreadblockShape 处理 SMEM 预算。 SM80/86 直接出现于设备与模板；tile/stages 必须根据目标设备资源重新选。
tags:
- grouped-gemm
- shared-memory
- tiling
kernel_types:
- grouped-gemm
symptoms:
- resource-limit
status: closed
confidence: source-reported
retrieved_at: '2026-09-17'
updated_at: '2025-01-24T21:16:29Z'
inclusion_reason: A100 上工作的 grouped GEMM 在 A10 上失败；讨论通过减小 ThreadblockShape 处理 SMEM 预算。
local_validation: not-run
---

# [QST] Grouped GEMM on A10 GPUs

[上游 ISSUE](https://github.com/NVIDIA/cutlass/issues/2053) · 核查于 2026-09-17 · **closed**

## 实现价值

A100 上工作的 grouped GEMM 在 A10 上失败；讨论通过减小 ThreadblockShape 处理 SMEM 预算。

## 使用边界

SM80/86 直接出现于设备与模板；tile/stages 必须根据目标设备资源重新选。

本地 GPU 验证：未运行。

## 讨论入口

- [讨论 2608428503](https://github.com/NVIDIA/cutlass/issues/2053#issuecomment-2608428503)（2025-01-22）
- [讨论 2613392747](https://github.com/NVIDIA/cutlass/issues/2053#issuecomment-2613392747)（2025-01-24）
