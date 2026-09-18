---
id: blog-siboehm-matmul-worklog
title: 'How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog'
type: blog
url: https://siboehm.com/articles/22/CUDA-MMM
architectures:
- sm80
- sm86
architecture_evidence:
  sm80: 主实验使用 Ampere RTX A6000（SM86），文中另有 A100（SM80）的配置比较。
  sm86: 主实验使用 Ampere RTX A6000（SM86），文中另有 A100（SM80）的配置比较。
tags:
- tiling
- coalescing
- vectorization
- register-blocking
kernel_types:
- gemm
symptoms:
- implementation-choice
status: published
confidence: source-reported
retrieved_at: '2026-09-17'
inclusion_reason: 作者逐步构建 FP32 SGEMM，适合研究合并访存、shared-memory tiling、寄存器分块和参数扫描。
local_validation: not-run
code_urls:
- https://github.com/siboehm/SGEMM_CUDA
---

# How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog

[原始来源](https://siboehm.com/articles/22/CUDA-MMM) · 核查于 2026-09-17

## 采用的内容

作者逐步构建 FP32 SGEMM，适合研究合并访存、shared-memory tiling、寄存器分块和参数扫描。

## 适用边界

这是 SIMT FP32 学习参考，不是 TensorOp 替代实现。边界条件与矩阵 shape 要另行验证；A6000/A100 的最优配置不同。

架构依据：主实验使用 Ampere RTX A6000（SM86），文中另有 A100（SM80）的配置比较。

本地 GPU 验证：未运行。

## 代码 / 实现入口

- [源码入口 1](https://github.com/siboehm/SGEMM_CUDA)
