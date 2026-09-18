---
id: doc-cuda-gpus
title: CUDA GPU Compute Capability
type: doc
url: https://developer.nvidia.com/cuda-gpus
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
inclusion_reason: 按准确 GPU 产品名查 compute capability；RTX A6000、RTX 6000 Ada、RTX PRO 6000 Blackwell 是不同架构。
local_validation: not-run
code_urls: []
---

# CUDA GPU Compute Capability

[原始来源](https://developer.nvidia.com/cuda-gpus) · 核查于 2026-09-17

## 采用的内容

按准确 GPU 产品名查 compute capability；RTX A6000、RTX 6000 Ada、RTX PRO 6000 Blackwell 是不同架构。

## 适用边界

在线文档会更新。采用当前版本前重新核查；不要将能力表理解为所有库 API 已实现。

架构依据：官方文档的对应 compute capability 条目；CuTe 通用概念的迁移不代表该 kernel 已在每张卡验证。

本地 GPU 验证：未运行。
