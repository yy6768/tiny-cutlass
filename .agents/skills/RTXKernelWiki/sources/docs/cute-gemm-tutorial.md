---
id: doc-cute-gemm-tutorial
title: CuTe GEMM Tutorial
type: doc
url: https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/media/docs/cpp/cute/0x_gemm_tutorial.md
architectures:
- sm80
- sm86
- sm89
architecture_evidence:
  sm80: 官方文档的对应 compute capability 条目；CuTe 通用概念的迁移不代表该 kernel 已在每张卡验证。
  sm86: 官方文档的对应 compute capability 条目；CuTe 通用概念的迁移不代表该 kernel 已在每张卡验证。
  sm89: 官方文档的对应 compute capability 条目；CuTe 通用概念的迁移不代表该 kernel 已在每张卡验证。
tags:
- cute
- tiling
- mma-sync
kernel_types:
- gemm
- attention
- convolution
symptoms:
- implementation-choice
status: published
confidence: source-reported
retrieved_at: '2026-09-17'
inclusion_reason: 官方教程提供线程/CTA 分块、TiledMMA 与 copy/GEMM 的组合入口。
local_validation: not-run
code_urls: []
---

# CuTe GEMM Tutorial

[原始来源](https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/media/docs/cpp/cute/0x_gemm_tutorial.md) · 核查于 2026-09-17

## 采用的内容

官方教程提供线程/CTA 分块、TiledMMA 与 copy/GEMM 的组合入口。

## 适用边界

在线文档会更新。采用当前版本前重新核查；不要将能力表理解为所有库 API 已实现。

架构依据：官方文档的对应 compute capability 条目；CuTe 通用概念的迁移不代表该 kernel 已在每张卡验证。

本地 GPU 验证：未运行。
