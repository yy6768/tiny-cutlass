---
id: issue-cutlass-2189
title: '[QST] Fusion paths on Ada/SM89'
type: issue
url: https://github.com/NVIDIA/cutlass/issues/2189
repo: NVIDIA/cutlass
number: 2189
architectures:
- sm89
architecture_evidence:
  sm89: 围绕 Ada implicit-GEMM convolution 讨论 mainloop、epilogue、back-to-back fusion 与 2.x EVT。 将此讨论作为 API 路线图；具体 conv
    类型与 visitor 可组合性仍看代码。
tags:
- convolution
- fusion
- evt
- epilogue
kernel_types:
- convolution
- gemm
symptoms:
- implementation-choice
status: closed
confidence: source-reported
retrieved_at: '2026-09-17'
updated_at: '2025-04-09T23:02:59Z'
inclusion_reason: 围绕 Ada implicit-GEMM convolution 讨论 mainloop、epilogue、back-to-back fusion 与 2.x EVT。
local_validation: not-run
---

# [QST] Fusion paths on Ada/SM89

[上游 ISSUE](https://github.com/NVIDIA/cutlass/issues/2189) · 核查于 2026-09-17 · **closed**

## 实现价值

围绕 Ada implicit-GEMM convolution 讨论 mainloop、epilogue、back-to-back fusion 与 2.x EVT。

## 使用边界

将此讨论作为 API 路线图；具体 conv 类型与 visitor 可组合性仍看代码。

本地 GPU 验证：未运行。

## 讨论入口

- [讨论 2767539407](https://github.com/NVIDIA/cutlass/issues/2189#issuecomment-2767539407)（2025-03-31）
