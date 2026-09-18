---
id: pr-triton-11262
title: '[NVIDIA] Support batched SM120 scaled-dot scale layouts'
type: pr
url: https://github.com/triton-lang/triton/pull/11262
repo: triton-lang/triton
number: 11262
architectures:
- sm120
architecture_evidence:
  sm120: 把 SM120 scaled-dot scale layout 转换扩展到 rank-3，保留 batch 维。 验证 batch 与矩阵维的处理、A/B 顺序和部署版本；不能仅因编译通过认定计算正确。
tags:
- block-scaling
- triton
- layout
kernel_types:
- gemm
- batched-gemm
symptoms:
- correctness
- compiler-crash
status: merged
confidence: source-reported
retrieved_at: '2026-09-17'
updated_at: '2026-08-11T20:57:04Z'
inclusion_reason: 把 SM120 scaled-dot scale layout 转换扩展到 rank-3，保留 batch 维。
local_validation: not-run
head_sha: a4792b9df48cabc58cef601d1ade6dc2db8f6343
merged_at: '2026-08-11T20:57:03Z'
merge_commit_sha: 161c6a272d54afe40f0386eade6743ec96577cc5
changed_files: 2
code_urls:
- https://github.com/chinazhangchao/triton-windows/blob/a4792b9df48cabc58cef601d1ade6dc2db8f6343/lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp
- https://github.com/chinazhangchao/triton-windows/blob/a4792b9df48cabc58cef601d1ade6dc2db8f6343/unittest/Dialect/TritonGPU/LinearLayoutConversionsTest.cpp
evidence_api: https://api.github.com/repos/triton-lang/triton/pulls/11262
---

# [NVIDIA] Support batched SM120 scaled-dot scale layouts

[上游 PR](https://github.com/triton-lang/triton/pull/11262) · 核查于 2026-09-17 · **merged**

## 实现价值

把 SM120 scaled-dot scale layout 转换扩展到 rank-3，保留 batch 维。

## 使用边界

验证 batch 与矩阵维的处理、A/B 顺序和部署版本；不能仅因编译通过认定计算正确。

本地 GPU 验证：未运行。

## 固定版本源码

PR head：`a4792b9df48cabc58cef601d1ade6dc2db8f6343`。共 2 个 changed files；以下是实现阅读入口，不是完整变更清单。

- [lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp](https://github.com/chinazhangchao/triton-windows/blob/a4792b9df48cabc58cef601d1ade6dc2db8f6343/lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp)
- [unittest/Dialect/TritonGPU/LinearLayoutConversionsTest.cpp](https://github.com/chinazhangchao/triton-windows/blob/a4792b9df48cabc58cef601d1ade6dc2db8f6343/unittest/Dialect/TritonGPU/LinearLayoutConversionsTest.cpp)
