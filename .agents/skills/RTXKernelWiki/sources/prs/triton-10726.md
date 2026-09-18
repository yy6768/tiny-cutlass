---
id: pr-triton-10726
title: '[SM120] Fallback MN-packed FP4 MMA to the decomposition path'
type: pr
url: https://github.com/triton-lang/triton/pull/10726
repo: triton-lang/triton
number: 10726
architectures:
- sm120
architecture_evidence:
  sm120: 为 SM120 MN-packed FP4 scaled dot 增加分解路径选择，避免送入要求 K packing 的 native MMA。 上游的显式分解路径不是本仓库 TensorOp 实验静默 fallback
    的授权；实现应明确支持边界。
tags:
- nvfp4
- packing
- triton
kernel_types:
- gemm
symptoms:
- compiler-crash
status: merged
confidence: source-reported
retrieved_at: '2026-09-17'
updated_at: '2026-07-07T13:21:37Z'
inclusion_reason: 为 SM120 MN-packed FP4 scaled dot 增加分解路径选择，避免送入要求 K packing 的 native MMA。
local_validation: not-run
head_sha: 6c06f28517fb7e798ce9c8357d650795c29a3cf2
merged_at: '2026-07-07T13:21:37Z'
merge_commit_sha: 2ddb39586b5ae5df266c0c31dedcd67e69301f59
changed_files: 2
code_urls:
- https://github.com/masahi/triton/blob/6c06f28517fb7e798ce9c8357d650795c29a3cf2/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp
- https://github.com/masahi/triton/blob/6c06f28517fb7e798ce9c8357d650795c29a3cf2/test/TritonGPU/accelerate-matmul.mlir
evidence_api: https://api.github.com/repos/triton-lang/triton/pulls/10726
---

# [SM120] Fallback MN-packed FP4 MMA to the decomposition path

[上游 PR](https://github.com/triton-lang/triton/pull/10726) · 核查于 2026-09-17 · **merged**

## 实现价值

为 SM120 MN-packed FP4 scaled dot 增加分解路径选择，避免送入要求 K packing 的 native MMA。

## 使用边界

上游的显式分解路径不是本仓库 TensorOp 实验静默 fallback 的授权；实现应明确支持边界。

本地 GPU 验证：未运行。

## 固定版本源码

PR head：`6c06f28517fb7e798ce9c8357d650795c29a3cf2`。共 2 个 changed files；以下是实现阅读入口，不是完整变更清单。

- [lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp](https://github.com/masahi/triton/blob/6c06f28517fb7e798ce9c8357d650795c29a3cf2/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp)
- [test/TritonGPU/accelerate-matmul.mlir](https://github.com/masahi/triton/blob/6c06f28517fb7e798ce9c8357d650795c29a3cf2/test/TritonGPU/accelerate-matmul.mlir)
