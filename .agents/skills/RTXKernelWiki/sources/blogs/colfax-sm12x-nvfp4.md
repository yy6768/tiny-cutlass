---
id: blog-colfax-sm12x-nvfp4
title: NVFP4 Blockscaled GEMM on NVIDIA RTX Pro Blackwell GPUs (SM12x)
type: blog
url: https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/
architectures:
- sm120
architecture_evidence:
  sm120: 正文直接覆盖 RTX PRO 6000 Blackwell（SM120），并链接 SM120 traits 与 Blackwell GeForce 示例。
tags:
- nvfp4
- block-scaling
- mma-sync
- tma
- cute-dsl
kernel_types:
- gemm
symptoms:
- implementation-choice
status: published
confidence: source-reported
retrieved_at: '2026-09-17'
inclusion_reason: 讲解 SM12x warp MMA 的 NVFP4 operand/scale fragment，以及从 dense GEMM 改成 blockscaled CuTe DSL kernel
  的步骤。
local_validation: not-run
code_urls:
- https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/include/cute/atom/mma_traits_sm120.hpp
- https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py
---

# NVFP4 Blockscaled GEMM on NVIDIA RTX Pro Blackwell GPUs (SM12x)

[原始来源](https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/) · 核查于 2026-09-17

## 采用的内容

讲解 SM12x warp MMA 的 NVFP4 operand/scale fragment，以及从 dense GEMM 改成 blockscaled CuTe DSL kernel 的步骤。

## 适用边界

明确区别 SM12x 与 SM10x 的 TMEM 模型。文章也讨论 SM121；只有与 SM120 对应的内容纳入。性能结果必须保留原 GPU、shape 和软件版本。

架构依据：正文直接覆盖 RTX PRO 6000 Blackwell（SM120），并链接 SM120 traits 与 Blackwell GeForce 示例。

本地 GPU 验证：未运行。

## 代码 / 实现入口

- [源码入口 1](https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/include/cute/atom/mma_traits_sm120.hpp)
- [源码入口 2](https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py)
