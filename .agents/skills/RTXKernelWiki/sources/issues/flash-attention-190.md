---
id: issue-flash-attention-190
title: Support for NVIDIA GeForce RTX 3090 with Compute Capability 8.6
type: issue
url: https://github.com/Dao-AILab/flash-attention/issues/190
repo: Dao-AILab/flash-attention
number: 190
architectures:
- sm86
architecture_evidence:
  sm86: 记录 RTX 3090 的早期 FlashAttention backward/head-dimension 限制及后续版本更新。 历史排查样本，不代表当前 head-dim 支持上限。使用当前 README、具体版本与
    reference 重新确认。
tags:
- attention
- shared-memory
kernel_types:
- attention
symptoms:
- resource-limit
- historical-support
status: closed
confidence: source-reported
retrieved_at: '2026-09-17'
updated_at: '2023-08-14T16:25:30Z'
inclusion_reason: 记录 RTX 3090 的早期 FlashAttention backward/head-dimension 限制及后续版本更新。
local_validation: not-run
---

# Support for NVIDIA GeForce RTX 3090 with Compute Capability 8.6

[上游 ISSUE](https://github.com/Dao-AILab/flash-attention/issues/190) · 核查于 2026-09-17 · **closed**

## 实现价值

记录 RTX 3090 的早期 FlashAttention backward/head-dimension 限制及后续版本更新。

## 使用边界

历史排查样本，不代表当前 head-dim 支持上限。使用当前 README、具体版本与 reference 重新确认。

本地 GPU 验证：未运行。

## 讨论入口

- [讨论 1639179612](https://github.com/Dao-AILab/flash-attention/issues/190#issuecomment-1639179612)（2023-07-18）
- [讨论 1639193815](https://github.com/Dao-AILab/flash-attention/issues/190#issuecomment-1639193815)（2023-07-18）
- [讨论 1639424910](https://github.com/Dao-AILab/flash-attention/issues/190#issuecomment-1639424910)（2023-07-18）
