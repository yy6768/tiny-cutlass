---
id: issue-cutlass-2158
title: '[DOC] CUTLASS INT4 GEMM: Missing SM89 Dispatch Configuration for L40S/4090'
type: issue
url: https://github.com/NVIDIA/cutlass/issues/2158
repo: NVIDIA/cutlass
number: 2158
architectures:
- sm80
- sm89
architecture_evidence:
  sm80: 维护者建议为 SM89 INT4 使用对应 SM80 kernel policy 并编译到 SM89。 不能由 SM89 测试文件名里缺少某个 dtype 推导硬件不支持；建议只适用于讨论的模板路径。
  sm89: 维护者建议为 SM89 INT4 使用对应 SM80 kernel policy 并编译到 SM89。 不能由 SM89 测试文件名里缺少某个 dtype 推导硬件不支持；建议只适用于讨论的模板路径。
tags:
- int4
- dispatch
- arch-tag
kernel_types:
- gemm
symptoms:
- compile-error
status: open
confidence: source-reported
retrieved_at: '2026-09-17'
updated_at: '2025-07-08T16:07:15Z'
inclusion_reason: 维护者建议为 SM89 INT4 使用对应 SM80 kernel policy 并编译到 SM89。
local_validation: not-run
---

# [DOC] CUTLASS INT4 GEMM: Missing SM89 Dispatch Configuration for L40S/4090

[上游 ISSUE](https://github.com/NVIDIA/cutlass/issues/2158) · 核查于 2026-09-17 · **open**

## 实现价值

维护者建议为 SM89 INT4 使用对应 SM80 kernel policy 并编译到 SM89。

## 使用边界

不能由 SM89 测试文件名里缺少某个 dtype 推导硬件不支持；建议只适用于讨论的模板路径。

本地 GPU 验证：未运行。

## 讨论入口

- [讨论 2710927406](https://github.com/NVIDIA/cutlass/issues/2158#issuecomment-2710927406)（2025-03-10）
