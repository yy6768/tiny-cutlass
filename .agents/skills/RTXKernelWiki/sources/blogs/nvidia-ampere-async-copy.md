---
id: blog-nvidia-ampere-async-copy
title: Controlling Data Movement to Boost Performance on the NVIDIA Ampere Architecture
type: blog
url: https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/
architectures:
- sm80
- sm86
architecture_evidence:
  sm80: 文章直接讨论 Ampere，并给出异步拷贝实现；SM80/86 资源差异另查官方 tuning guide。
  sm86: 文章直接讨论 Ampere，并给出异步拷贝实现；SM80/86 资源差异另查官方 tuning guide。
tags:
- cp-async
- pipeline
- shared-memory
kernel_types:
- gemm
- convolution
symptoms:
- implementation-choice
status: published
confidence: source-reported
retrieved_at: '2026-09-17'
inclusion_reason: 官方示例从同步加载推进到 cuda::memcpy_async、barrier 和多阶段 pipeline，可用于理解 global→shared 与计算重叠。
local_validation: not-run
code_urls: []
---

# Controlling Data Movement to Boost Performance on the NVIDIA Ampere Architecture

[原始来源](https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/) · 核查于 2026-09-17

## 采用的内容

官方示例从同步加载推进到 cuda::memcpy_async、barrier 和多阶段 pipeline，可用于理解 global→shared 与计算重叠。

## 适用边界

原文对应 CUDA 11.1；同步、对齐和 API 写法须对照当前 CUDA。它没有提供 Ada/SM120 的实测，不能推导统一加速比。

架构依据：文章直接讨论 Ampere，并给出异步拷贝实现；SM80/86 资源差异另查官方 tuning guide。

本地 GPU 验证：未运行。
