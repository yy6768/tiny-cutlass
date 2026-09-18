---
id: blog-kapil-ada-cutlass
title: Learn CUTLASS the hard way!
type: blog
url: https://www.kapilsharma.dev/posts/learn-cutlass-the-hard-way/
architectures:
- sm89
architecture_evidence:
  sm89: 文章明确声明 benchmark 设备为 RTX 4090；博客中的编译 API/性能陈述仍是版本化的作者报告。
tags:
- mma-sync
- cutlass2
- cp-async
- tiling
kernel_types:
- gemm
symptoms:
- implementation-choice
status: published
confidence: source-reported
retrieved_at: '2026-09-17'
inclusion_reason: RTX 4090 上从 FP32 分块逐步走到 FP16/BF16 Tensor Core、流水和 CUTLASS 2.x 配置，可对照作者源码学习。
local_validation: not-run
code_urls:
- https://github.com/gpusgobrr/explore-gemm
---

# Learn CUTLASS the hard way!

[原始来源](https://www.kapilsharma.dev/posts/learn-cutlass-the-hard-way/) · 核查于 2026-09-17

## 采用的内容

RTX 4090 上从 FP32 分块逐步走到 FP16/BF16 Tensor Core、流水和 CUTLASS 2.x 配置，可对照作者源码学习。

## 适用边界

限用 Ada/CUTLASS 实现部分。原文混淆了 L1/shared 总容量与可用 SMEM，并给出不准确的 resident-block 数；硬件值以 NVIDIA guide/runtime 为准。后续仓库新增 Hopper 文件不在此条覆盖内。

架构依据：文章明确声明 benchmark 设备为 RTX 4090；博客中的编译 API/性能陈述仍是版本化的作者报告。

本地 GPU 验证：未运行。

## 代码 / 实现入口

- [源码入口 1](https://github.com/gpusgobrr/explore-gemm)
