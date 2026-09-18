---
id: issue-cutlass-3281
title: '[BUG] CuTeDSL 4.5.2: MmaFP8Op.make_fragment_A segfaults on SM120 (Blackwell)'
type: issue
url: https://github.com/NVIDIA/cutlass/issues/3281
repo: NVIDIA/cutlass
number: 3281
architectures:
- sm120
architecture_evidence:
  sm120: CuTe DSL 4.5.2 在 RTX 5090 上使用 MmaFP8Op.make_fragment 时的 crash reproducer。 根因是报告者解释，尚无本地验证。检查当前 warp atom
    对 SM120 的支持，不只照抄类名。
tags:
- fp8
- cute-dsl
- mma-sync
kernel_types:
- gemm
symptoms:
- compiler-crash
status: open
confidence: source-reported
retrieved_at: '2026-09-17'
updated_at: '2026-06-28T03:15:43Z'
inclusion_reason: CuTe DSL 4.5.2 在 RTX 5090 上使用 MmaFP8Op.make_fragment 时的 crash reproducer。
local_validation: not-run
---

# [BUG] CuTeDSL 4.5.2: MmaFP8Op.make_fragment_A segfaults on SM120 (Blackwell)

[上游 ISSUE](https://github.com/NVIDIA/cutlass/issues/3281) · 核查于 2026-09-17 · **open**

## 实现价值

CuTe DSL 4.5.2 在 RTX 5090 上使用 MmaFP8Op.make_fragment 时的 crash reproducer。

## 使用边界

根因是报告者解释，尚无本地验证。检查当前 warp atom 对 SM120 的支持，不只照抄类名。

本地 GPU 验证：未运行。

## 讨论入口

- [讨论 4570088319](https://github.com/NVIDIA/cutlass/issues/3281#issuecomment-4570088319)（2026-05-29）
