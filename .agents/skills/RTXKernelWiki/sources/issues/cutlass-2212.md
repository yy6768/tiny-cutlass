---
id: issue-cutlass-2212
title: '[QST]Why cp.async.ca will influence bank conflcit of Shared Store From Global Load ?'
type: issue
url: https://github.com/NVIDIA/cutlass/issues/2212
repo: NVIDIA/cutlass
number: 2212
architectures:
- sm86
- sm89
architecture_evidence:
  sm86: 提供 cp.async.ca/cg 与 swizzle、bank conflict 之间关系的复现讨论，涉及 RTX 3060/4060 Ti。 评论包含明确标注的硬件机理猜测；仅作为实验线索。用目标卡 NCU
    与实际访问模式确认，不当作确定的 cache 规则。
  sm89: 提供 cp.async.ca/cg 与 swizzle、bank conflict 之间关系的复现讨论，涉及 RTX 3060/4060 Ti。 评论包含明确标注的硬件机理猜测；仅作为实验线索。用目标卡 NCU
    与实际访问模式确认，不当作确定的 cache 规则。
tags:
- cp-async
- swizzle
- convolution
kernel_types:
- convolution
symptoms:
- bank-conflict
status: open
confidence: source-reported
retrieved_at: '2026-09-17'
updated_at: '2026-01-30T05:20:49Z'
inclusion_reason: 提供 cp.async.ca/cg 与 swizzle、bank conflict 之间关系的复现讨论，涉及 RTX 3060/4060 Ti。
local_validation: not-run
---

# [QST]Why cp.async.ca will influence bank conflcit of Shared Store From Global Load ?

[上游 ISSUE](https://github.com/NVIDIA/cutlass/issues/2212) · 核查于 2026-09-17 · **open**

## 实现价值

提供 cp.async.ca/cg 与 swizzle、bank conflict 之间关系的复现讨论，涉及 RTX 3060/4060 Ti。

## 使用边界

评论包含明确标注的硬件机理猜测；仅作为实验线索。用目标卡 NCU 与实际访问模式确认，不当作确定的 cache 规则。

本地 GPU 验证：未运行。

## 讨论入口

- [讨论 3471742175](https://github.com/NVIDIA/cutlass/issues/2212#issuecomment-3471742175)（2025-10-31）
- [讨论 3817944848](https://github.com/NVIDIA/cutlass/issues/2212#issuecomment-3817944848)（2026-01-29）
- [讨论 3821491536](https://github.com/NVIDIA/cutlass/issues/2212#issuecomment-3821491536)（2026-01-30）
