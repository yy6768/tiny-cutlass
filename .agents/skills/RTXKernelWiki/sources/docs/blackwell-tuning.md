---
id: doc-blackwell-tuning
title: NVIDIA Blackwell Tuning Guide
type: doc
url: https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
architectures:
- sm120
architecture_evidence:
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
inclusion_reason: 查 CC 12.0 对应的资源与编程条件，逐段区别 CC 10.0。
local_validation: not-run
code_urls: []
---

# NVIDIA Blackwell Tuning Guide

[原始来源](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) · 核查于 2026-09-17

## 采用的内容

查 CC 12.0 对应的资源与编程条件，逐段区别 CC 10.0。

## 适用边界

在线文档会更新。采用当前版本前重新核查；不要将能力表理解为所有库 API 已实现。

架构依据：官方文档的对应 compute capability 条目；CuTe 通用概念的迁移不代表该 kernel 已在每张卡验证。

本地 GPU 验证：未运行。
