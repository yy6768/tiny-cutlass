---
id: issue-cutlass-2906
title: '[BUG] SM120 NVF4 GEMM (example 79a): misaligned address crash'
type: issue
url: https://github.com/NVIDIA/cutlass/issues/2906
repo: NVIDIA/cutlass
number: 2906
architectures:
- sm120
architecture_evidence:
  sm120: 记录 Windows 11、CUDA 13.1、CUTLASS 4.3.4 下 example 79a 的对齐故障。 重点核对 TMA descriptor 和 scale 的 alignment/ABI；后续讨论仍在跟踪
    MSVC 与更新 CUDA，不当作统一修复。
tags:
- nvfp4
- tma
- alignment
- windows
kernel_types:
- gemm
symptoms:
- misaligned-address
status: open
confidence: source-reported
retrieved_at: '2026-09-17'
updated_at: '2026-05-30T00:58:35Z'
inclusion_reason: 记录 Windows 11、CUDA 13.1、CUTLASS 4.3.4 下 example 79a 的对齐故障。
local_validation: not-run
---

# [BUG] SM120 NVF4 GEMM (example 79a): misaligned address crash

[上游 ISSUE](https://github.com/NVIDIA/cutlass/issues/2906) · 核查于 2026-09-17 · **open**

## 实现价值

记录 Windows 11、CUDA 13.1、CUTLASS 4.3.4 下 example 79a 的对齐故障。

## 使用边界

重点核对 TMA descriptor 和 scale 的 alignment/ABI；后续讨论仍在跟踪 MSVC 与更新 CUDA，不当作统一修复。

本地 GPU 验证：未运行。

## 讨论入口

- [讨论 3725339144](https://github.com/NVIDIA/cutlass/issues/2906#issuecomment-3725339144)（2026-01-08）
- [讨论 3729548602](https://github.com/NVIDIA/cutlass/issues/2906#issuecomment-3729548602)（2026-01-09）
- [讨论 4580990742](https://github.com/NVIDIA/cutlass/issues/2906#issuecomment-4580990742)（2026-05-30）
