---
id: blog-colfax-cute-transpose
title: 'Tutorial: Matrix Transpose in CUTLASS'
type: blog
url: https://research.colfax-intl.com/tutorial-matrix-transpose-in-cutlass/
architectures:
- sm80
- sm86
- sm89
- sm120
architecture_evidence:
  sm80: 这是 portable layout 原理的迁移推断，不是四种架构上的 kernel 支持证明。
  sm86: 这是 portable layout 原理的迁移推断，不是四种架构上的 kernel 支持证明。
  sm89: 这是 portable layout 原理的迁移推断，不是四种架构上的 kernel 支持证明。
  sm120: 这是 portable layout 原理的迁移推断，不是四种架构上的 kernel 支持证明。
tags:
- cute
- layout
- swizzle
- coalescing
kernel_types:
- transpose
- gemm
- convolution
symptoms:
- implementation-choice
status: published
confidence: inferred
retrieved_at: '2026-09-17'
inclusion_reason: 选取基础 CuTe layout、TiledCopy 与 shared-memory swizzle 的推导，用于分析线程到数据的映射。
local_validation: not-run
code_urls: []
---

# Tutorial: Matrix Transpose in CUTLASS

[原始来源](https://research.colfax-intl.com/tutorial-matrix-transpose-in-cutlass/) · 核查于 2026-09-17

## 采用的内容

选取基础 CuTe layout、TiledCopy 与 shared-memory swizzle 的推导，用于分析线程到数据的映射。

## 适用边界

跨架构标签仅表示通用 layout 部分可迁移。文章后半有 Hopper TMA 与对应测量；SM80/86/89 不采用 TMA 代码，SM120 需另查其 TMA 指令与示例。没有目标卡全覆盖实测。

架构依据：这是 portable layout 原理的迁移推断，不是四种架构上的 kernel 支持证明。

本地 GPU 验证：未运行。
