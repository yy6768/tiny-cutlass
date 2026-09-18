---
id: issue-triton-11733
title: '3.8.0 on sm_120: MN-packed mxfp4 dot_scaled returns wrong results and batched scaled dots fail an assert;
  both fixed on main (#10726, #11262)'
type: issue
url: https://github.com/triton-lang/triton/issues/11733
repo: triton-lang/triton
number: 11733
architectures:
- sm120
architecture_evidence:
  sm120: '在 Triton 3.8.0/RTX PRO 6000 报告 MN-packed FP4 错误输出和 batched scale layout 失败。 与 main 上 #10726/#11262 区分；main
    合并不自动表示 release wheel 包含修复。'
tags:
- nvfp4
- triton
- block-scaling
kernel_types:
- gemm
symptoms:
- correctness
- compiler-crash
status: open
confidence: source-reported
retrieved_at: '2026-09-17'
updated_at: '2026-09-12T09:26:48Z'
inclusion_reason: 在 Triton 3.8.0/RTX PRO 6000 报告 MN-packed FP4 错误输出和 batched scale layout 失败。
local_validation: not-run
---

# 3.8.0 on sm_120: MN-packed mxfp4 dot_scaled returns wrong results and batched scaled dots fail an assert; both fixed on main (#10726, #11262)

[上游 ISSUE](https://github.com/triton-lang/triton/issues/11733) · 核查于 2026-09-17 · **open**

## 实现价值

在 Triton 3.8.0/RTX PRO 6000 报告 MN-packed FP4 错误输出和 batched scale layout 失败。

## 使用边界

与 main 上 #10726/#11262 区分；main 合并不自动表示 release wheel 包含修复。

本地 GPU 验证：未运行。
