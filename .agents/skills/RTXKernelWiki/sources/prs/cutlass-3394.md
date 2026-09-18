---
id: pr-cutlass-3394
title: Fix SM89 FP8 blockwise scale indexing under threadblock swizzling
type: pr
url: https://github.com/NVIDIA/cutlass/pull/3394
repo: NVIDIA/cutlass
number: 3394
architectures:
- sm89
architecture_evidence:
  sm89: 将 blockwise scale 的索引从原始 blockIdx 改为 swizzle 后的逻辑 M/N tile 坐标。 验证不同 swizzle、非整 tile 和 exact-size scale tensor；不要只验证默认
    swizzle。
tags:
- fp8
- block-scaling
- swizzle
kernel_types:
- gemm
symptoms:
- correctness
- illegal-memory-access
status: merged
confidence: source-reported
retrieved_at: '2026-09-17'
updated_at: '2026-08-06T02:01:32Z'
inclusion_reason: 将 blockwise scale 的索引从原始 blockIdx 改为 swizzle 后的逻辑 M/N tile 坐标。
local_validation: not-run
head_sha: 3c43992d8c5ff2f55b61d638a58e14421a7c12fe
merged_at: '2026-08-06T02:01:31Z'
merge_commit_sha: b50d8fd7196e1c5801c1bd8edf1d64e4bf9add6c
changed_files: 4
code_urls:
- https://github.com/Noperi0r/cutlass/blob/3c43992d8c5ff2f55b61d638a58e14421a7c12fe/include/cutlass/gemm/kernel/gemm_universal_blockwise.h
- https://github.com/Noperi0r/cutlass/blob/3c43992d8c5ff2f55b61d638a58e14421a7c12fe/include/cutlass/gemm/threadblock/mma_multistage_blockwise.h
- https://github.com/Noperi0r/cutlass/blob/3c43992d8c5ff2f55b61d638a58e14421a7c12fe/test/unit/gemm/device/gemm_f8t_f8n_bf16t_tensor_op_f32_blockwise_sm89.cu
evidence_api: https://api.github.com/repos/NVIDIA/cutlass/pulls/3394
---

# Fix SM89 FP8 blockwise scale indexing under threadblock swizzling

[上游 PR](https://github.com/NVIDIA/cutlass/pull/3394) · 核查于 2026-09-17 · **merged**

## 实现价值

将 blockwise scale 的索引从原始 blockIdx 改为 swizzle 后的逻辑 M/N tile 坐标。

## 使用边界

验证不同 swizzle、非整 tile 和 exact-size scale tensor；不要只验证默认 swizzle。

本地 GPU 验证：未运行。

## 固定版本源码

PR head：`3c43992d8c5ff2f55b61d638a58e14421a7c12fe`。共 4 个 changed files；以下是实现阅读入口，不是完整变更清单。

- [include/cutlass/gemm/kernel/gemm_universal_blockwise.h](https://github.com/Noperi0r/cutlass/blob/3c43992d8c5ff2f55b61d638a58e14421a7c12fe/include/cutlass/gemm/kernel/gemm_universal_blockwise.h)
- [include/cutlass/gemm/threadblock/mma_multistage_blockwise.h](https://github.com/Noperi0r/cutlass/blob/3c43992d8c5ff2f55b61d638a58e14421a7c12fe/include/cutlass/gemm/threadblock/mma_multistage_blockwise.h)
- [test/unit/gemm/device/gemm_f8t_f8n_bf16t_tensor_op_f32_blockwise_sm89.cu](https://github.com/Noperi0r/cutlass/blob/3c43992d8c5ff2f55b61d638a58e14421a7c12fe/test/unit/gemm/device/gemm_f8t_f8n_bf16t_tensor_op_f32_blockwise_sm89.cu)

## 讨论入口

- [讨论 5178652637](https://github.com/NVIDIA/cutlass/pull/3394#issuecomment-5178652637)（2026-08-04）
