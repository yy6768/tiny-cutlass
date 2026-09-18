---
id: doc-ampere-tuning
title: NVIDIA Ampere GPU Architecture Tuning Guide
type: doc
url: https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
architectures:
- sm80
- sm86
architecture_evidence:
  sm80: 官方文档的对应 compute capability 条目；CuTe 通用概念的迁移不代表该 kernel 已在每张卡验证。
  sm86: 官方文档的对应 compute capability 条目；CuTe 通用概念的迁移不代表该 kernel 已在每张卡验证。
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
inclusion_reason: 查询 CC 8.0/8.6 的 occupancy、shared-memory、异步拷贝和 Tensor Core 差异。
local_validation: not-run
code_urls: []
---

# NVIDIA Ampere GPU Architecture Tuning Guide

[原始来源](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html) · 核查于 2026-09-17

## 采用的内容

查询 CC 8.0/8.6 的 occupancy、shared-memory、异步拷贝和 Tensor Core 差异。

## 适用边界

在线文档会更新。采用当前版本前重新核查；不要将能力表理解为所有库 API 已实现。

架构依据：官方文档的对应 compute capability 条目；CuTe 通用概念的迁移不代表该 kernel 已在每张卡验证。

本地 GPU 验证：未运行。
