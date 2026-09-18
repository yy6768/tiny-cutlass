---
id: doc-ptx-isa
title: NVIDIA PTX ISA — instruction target notes
type: doc
url: https://docs.nvidia.com/cuda/parallel-thread-execution/
architectures:
- sm80
- sm86
- sm89
- sm120
architecture_evidence:
  sm80: 官方文档的对应 compute capability 条目；CuTe 通用概念的迁移不代表该 kernel 已在每张卡验证。
  sm86: 官方文档的对应 compute capability 条目；CuTe 通用概念的迁移不代表该 kernel 已在每张卡验证。
  sm89: 官方文档的对应 compute capability 条目；CuTe 通用概念的迁移不代表该 kernel 已在每张卡验证。
  sm120: 官方文档的对应 compute capability 条目；CuTe 通用概念的迁移不代表该 kernel 已在每张卡验证。
tags:
- architecture
- documentation
kernel_types:
- gemm
- attention
- convolution
symptoms:
- implementation-choice
status: published
confidence: source-reported
retrieved_at: '2026-09-17'
inclusion_reason: 按指令 Target ISA Notes 确认 cp.async、ldmatrix、mma、TMA、低精度转换及 a/f 目标要求。
local_validation: not-run
code_urls: []
---

# NVIDIA PTX ISA — instruction target notes

[原始来源](https://docs.nvidia.com/cuda/parallel-thread-execution/) · 核查于 2026-09-17

## 采用的内容

按指令 Target ISA Notes 确认 cp.async、ldmatrix、mma、TMA、低精度转换及 a/f 目标要求。

## 适用边界

在线文档会更新。采用当前版本前重新核查；不要将能力表理解为所有库 API 已实现。

架构依据：官方文档的对应 compute capability 条目；CuTe 通用概念的迁移不代表该 kernel 已在每张卡验证。

本地 GPU 验证：未运行。
