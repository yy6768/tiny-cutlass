---
id: pr-cutlass-3030
title: '[CuTeDSL] Flash Attention v2 for SM120 (Blackwell GeForce)'
type: pr
url: https://github.com/NVIDIA/cutlass/pull/3030
repo: NVIDIA/cutlass
number: 3030
architectures:
- sm120
architecture_evidence:
  sm120: SM120 FA2 forward 候选同时提供 cp.async 和 TMA pipeline，使用 warp MMA。 open PR；较新的讨论报告了 GB10 测试并更正早期数据。SM121 不等于
    SM120，不引用旧性能表证明 RTX 优势。
tags:
- attention
- tma
- cp-async
- cute-dsl
- fp8
kernel_types:
- attention
symptoms:
- resource-limit
- implementation-choice
status: open
confidence: source-reported
retrieved_at: '2026-09-17'
updated_at: '2026-08-14T22:02:01Z'
inclusion_reason: SM120 FA2 forward 候选同时提供 cp.async 和 TMA pipeline，使用 warp MMA。
local_validation: not-run
head_sha: a3402a18a6e5e0d80d2c65fce540fbe8f278d317
merged_at: null
merge_commit_sha: 859d8a3e8212405bf493ace2f54b3b2461c7f739
changed_files: 5
code_urls:
- https://github.com/blake-snc/cutlass/blob/a3402a18a6e5e0d80d2c65fce540fbe8f278d317/examples/python/CuTeDSL/blackwell_geforce/benchmark_fp8_vs_bf16.py
- https://github.com/blake-snc/cutlass/blob/a3402a18a6e5e0d80d2c65fce540fbe8f278d317/examples/python/CuTeDSL/blackwell_geforce/flash_attention_v2.py
- https://github.com/blake-snc/cutlass/blob/a3402a18a6e5e0d80d2c65fce540fbe8f278d317/examples/python/CuTeDSL/blackwell_geforce/fp8_flash_attention.py
- https://github.com/blake-snc/cutlass/blob/a3402a18a6e5e0d80d2c65fce540fbe8f278d317/examples/python/CuTeDSL/blackwell_geforce/fp8_flash_attention_tma.py
- https://github.com/blake-snc/cutlass/blob/a3402a18a6e5e0d80d2c65fce540fbe8f278d317/examples/python/CuTeDSL/blackwell_geforce/fp8_gemm.py
evidence_api: https://api.github.com/repos/NVIDIA/cutlass/pulls/3030
---

# [CuTeDSL] Flash Attention v2 for SM120 (Blackwell GeForce)

[上游 PR](https://github.com/NVIDIA/cutlass/pull/3030) · 核查于 2026-09-17 · **open**

## 实现价值

SM120 FA2 forward 候选同时提供 cp.async 和 TMA pipeline，使用 warp MMA。

## 使用边界

open PR；较新的讨论报告了 GB10 测试并更正早期数据。SM121 不等于 SM120，不引用旧性能表证明 RTX 优势。

本地 GPU 验证：未运行。

## 固定版本源码

PR head：`a3402a18a6e5e0d80d2c65fce540fbe8f278d317`。共 5 个 changed files；以下是实现阅读入口，不是完整变更清单。

- [examples/python/CuTeDSL/blackwell_geforce/benchmark_fp8_vs_bf16.py](https://github.com/blake-snc/cutlass/blob/a3402a18a6e5e0d80d2c65fce540fbe8f278d317/examples/python/CuTeDSL/blackwell_geforce/benchmark_fp8_vs_bf16.py)
- [examples/python/CuTeDSL/blackwell_geforce/flash_attention_v2.py](https://github.com/blake-snc/cutlass/blob/a3402a18a6e5e0d80d2c65fce540fbe8f278d317/examples/python/CuTeDSL/blackwell_geforce/flash_attention_v2.py)
- [examples/python/CuTeDSL/blackwell_geforce/fp8_flash_attention.py](https://github.com/blake-snc/cutlass/blob/a3402a18a6e5e0d80d2c65fce540fbe8f278d317/examples/python/CuTeDSL/blackwell_geforce/fp8_flash_attention.py)
- [examples/python/CuTeDSL/blackwell_geforce/fp8_flash_attention_tma.py](https://github.com/blake-snc/cutlass/blob/a3402a18a6e5e0d80d2c65fce540fbe8f278d317/examples/python/CuTeDSL/blackwell_geforce/fp8_flash_attention_tma.py)
- [examples/python/CuTeDSL/blackwell_geforce/fp8_gemm.py](https://github.com/blake-snc/cutlass/blob/a3402a18a6e5e0d80d2c65fce540fbe8f278d317/examples/python/CuTeDSL/blackwell_geforce/fp8_gemm.py)

## 讨论入口

- [讨论 4637838547](https://github.com/NVIDIA/cutlass/pull/3030#issuecomment-4637838547)（2026-06-06）
- [讨论 4703766858](https://github.com/NVIDIA/cutlass/pull/3030#issuecomment-4703766858)（2026-06-15）
- [讨论 5298603275](https://github.com/NVIDIA/cutlass/pull/3030#issuecomment-5298603275)（2026-08-14）
