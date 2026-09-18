---
id: pr-flashinfer-2786
title: 'feat: K=64 block-scaled MoE GEMM for SM120 (RTX PRO 6000)'
type: pr
url: https://github.com/flashinfer-ai/flashinfer/pull/2786
repo: flashinfer-ai/flashinfer
number: 2786
architectures:
- sm120
architecture_evidence:
  sm120: K=64 blockscaled MoE 候选展示 scale layout、SMEM 与 TMA 的交互。 closed-unmerged，依赖作者 CUTLASS 分支；正文的无可用 tile、性能结论不可泛化。
tags:
- nvfp4
- tiling
- grouped-gemm
kernel_types:
- grouped-gemm
- moe
symptoms:
- resource-limit
status: closed-unmerged
confidence: source-reported
retrieved_at: '2026-09-17'
updated_at: '2026-03-29T04:31:18Z'
inclusion_reason: K=64 blockscaled MoE 候选展示 scale layout、SMEM 与 TMA 的交互。
local_validation: not-run
head_sha: 2449aadaaef0a4a78bdcbb9b1ae4148378ba8bdb
merged_at: null
merge_commit_sha: null
changed_files: 4
code_urls:
- https://github.com/brandonmmusic-max/flashinfer/blob/2449aadaaef0a4a78bdcbb9b1ae4148378ba8bdb/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch_tma_ws.h
- https://github.com/brandonmmusic-max/flashinfer/blob/2449aadaaef0a4a78bdcbb9b1ae4148378ba8bdb/flashinfer/jit/gemm/cutlass/generate_kernels.py
evidence_api: https://api.github.com/repos/flashinfer-ai/flashinfer/pulls/2786
---

# feat: K=64 block-scaled MoE GEMM for SM120 (RTX PRO 6000)

[上游 PR](https://github.com/flashinfer-ai/flashinfer/pull/2786) · 核查于 2026-09-17 · **closed-unmerged**

## 实现价值

K=64 blockscaled MoE 候选展示 scale layout、SMEM 与 TMA 的交互。

## 使用边界

closed-unmerged，依赖作者 CUTLASS 分支；正文的无可用 tile、性能结论不可泛化。

本地 GPU 验证：未运行。

## 固定版本源码

PR head：`2449aadaaef0a4a78bdcbb9b1ae4148378ba8bdb`。共 4 个 changed files；以下是实现阅读入口，不是完整变更清单。

- [csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch_tma_ws.h](https://github.com/brandonmmusic-max/flashinfer/blob/2449aadaaef0a4a78bdcbb9b1ae4148378ba8bdb/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch_tma_ws.h)
- [flashinfer/jit/gemm/cutlass/generate_kernels.py](https://github.com/brandonmmusic-max/flashinfer/blob/2449aadaaef0a4a78bdcbb9b1ae4148378ba8bdb/flashinfer/jit/gemm/cutlass/generate_kernels.py)

## 讨论入口

- [讨论 4101899526](https://github.com/flashinfer-ai/flashinfer/pull/2786#issuecomment-4101899526)（2026-03-21）
- [讨论 4102361160](https://github.com/flashinfer-ai/flashinfer/pull/2786#issuecomment-4102361160)（2026-03-21）
- [讨论 4149408606](https://github.com/flashinfer-ai/flashinfer/pull/2786#issuecomment-4149408606)（2026-03-29）
