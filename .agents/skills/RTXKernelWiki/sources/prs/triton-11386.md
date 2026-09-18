---
id: pr-triton-11386
title: '[NVIDIA] Use block-scaled MMA with unit scales for fp8 dot on sm120'
type: pr
url: https://github.com/triton-lang/triton/pull/11386
repo: triton-lang/triton
number: 11386
architectures:
- sm120
architecture_evidence:
  sm120: 提出用单位 scale 的 blockscaled MMA 替换部分 SM120 FP8 dot。 closed-unmerged；吞吐机理仍有争论，仅作实验候选，不写成编译器已采用或 NVIDIA 官方保证。
tags:
- fp8
- block-scaling
- triton
kernel_types:
- gemm
symptoms:
- throughput
status: closed-unmerged
confidence: source-reported
retrieved_at: '2026-09-17'
updated_at: '2026-08-21T09:11:12Z'
inclusion_reason: 提出用单位 scale 的 blockscaled MMA 替换部分 SM120 FP8 dot。
local_validation: not-run
head_sha: fcf734bad15ddbe5e2955a6f76e50204b9f48aeb
merged_at: null
merge_commit_sha: fb747b0ccb81a040a96e96ef44ba53143b06f3fe
changed_files: 4
code_urls:
- https://github.com/SamMausberg/triton/blob/fcf734bad15ddbe5e2955a6f76e50204b9f48aeb/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp
- https://github.com/SamMausberg/triton/blob/fcf734bad15ddbe5e2955a6f76e50204b9f48aeb/python/test/unit/language/test_core.py
- https://github.com/SamMausberg/triton/blob/fcf734bad15ddbe5e2955a6f76e50204b9f48aeb/test/Conversion/tritongpu_to_llvm_sm120.mlir
- https://github.com/SamMausberg/triton/blob/fcf734bad15ddbe5e2955a6f76e50204b9f48aeb/test/TritonGPU/accelerate-matmul.mlir
evidence_api: https://api.github.com/repos/triton-lang/triton/pulls/11386
---

# [NVIDIA] Use block-scaled MMA with unit scales for fp8 dot on sm120

[上游 PR](https://github.com/triton-lang/triton/pull/11386) · 核查于 2026-09-17 · **closed-unmerged**

## 实现价值

提出用单位 scale 的 blockscaled MMA 替换部分 SM120 FP8 dot。

## 使用边界

closed-unmerged；吞吐机理仍有争论，仅作实验候选，不写成编译器已采用或 NVIDIA 官方保证。

本地 GPU 验证：未运行。

## 固定版本源码

PR head：`fcf734bad15ddbe5e2955a6f76e50204b9f48aeb`。共 4 个 changed files；以下是实现阅读入口，不是完整变更清单。

- [lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp](https://github.com/SamMausberg/triton/blob/fcf734bad15ddbe5e2955a6f76e50204b9f48aeb/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp)
- [python/test/unit/language/test_core.py](https://github.com/SamMausberg/triton/blob/fcf734bad15ddbe5e2955a6f76e50204b9f48aeb/python/test/unit/language/test_core.py)
- [test/Conversion/tritongpu_to_llvm_sm120.mlir](https://github.com/SamMausberg/triton/blob/fcf734bad15ddbe5e2955a6f76e50204b9f48aeb/test/Conversion/tritongpu_to_llvm_sm120.mlir)
- [test/TritonGPU/accelerate-matmul.mlir](https://github.com/SamMausberg/triton/blob/fcf734bad15ddbe5e2955a6f76e50204b9f48aeb/test/TritonGPU/accelerate-matmul.mlir)

## 讨论入口

- [讨论 5364601229](https://github.com/triton-lang/triton/pull/11386#issuecomment-5364601229)（2026-08-21）
- [讨论 5367848515](https://github.com/triton-lang/triton/pull/11386#issuecomment-5367848515)（2026-08-21）
- [讨论 5367912781](https://github.com/triton-lang/triton/pull/11386#issuecomment-5367912781)（2026-08-21）
