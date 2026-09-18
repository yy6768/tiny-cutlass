---
id: pr-cutlass-2328
title: Add SM80/89 blockwise scaling kernel, support FP8 block/groupwise on Ada, INT8 on Ampere
type: pr
url: https://github.com/NVIDIA/cutlass/pull/2328
repo: NVIDIA/cutlass
number: 2328
architectures:
- sm80
- sm86
- sm89
architecture_evidence:
  sm80: 提出 Ampere INT8 与 Ada FP8 的 block/groupwise scaling CuTe 路径；正文包含 A100、A10 和 4090。 仍是 open PR；FP8 对应 SM89，INT8
    对应 Ampere。不要把同页多架构标签理解成 Ampere 支持原生 FP8。
  sm86: 提出 Ampere INT8 与 Ada FP8 的 block/groupwise scaling CuTe 路径；正文包含 A100、A10 和 4090。 仍是 open PR；FP8 对应 SM89，INT8
    对应 Ampere。不要把同页多架构标签理解成 Ampere 支持原生 FP8。
  sm89: 提出 Ampere INT8 与 Ada FP8 的 block/groupwise scaling CuTe 路径；正文包含 A100、A10 和 4090。 仍是 open PR；FP8 对应 SM89，INT8
    对应 Ampere。不要把同页多架构标签理解成 Ampere 支持原生 FP8。
tags:
- fp8
- int8
- block-scaling
- cutlass3
kernel_types:
- gemm
symptoms:
- implementation-choice
status: open
confidence: source-reported
retrieved_at: '2026-09-17'
updated_at: '2026-02-10T11:22:59Z'
inclusion_reason: 提出 Ampere INT8 与 Ada FP8 的 block/groupwise scaling CuTe 路径；正文包含 A100、A10 和 4090。
local_validation: not-run
head_sha: 30ac8288731b1180cf6cb483d85948e5efc046ce
merged_at: null
merge_commit_sha: 7356b538ed5d21468eb9ac51b6f46781c714a4be
changed_files: 11
code_urls:
- https://github.com/solrex/cutlass/blob/30ac8288731b1180cf6cb483d85948e5efc046ce/include/cutlass/gemm/collective/collective_mma.hpp
- https://github.com/solrex/cutlass/blob/30ac8288731b1180cf6cb483d85948e5efc046ce/include/cutlass/gemm/collective/sm80_mma_multistage_blockwise_scaling.hpp
- https://github.com/solrex/cutlass/blob/30ac8288731b1180cf6cb483d85948e5efc046ce/include/cutlass/gemm/dispatch_policy.hpp
- https://github.com/solrex/cutlass/blob/30ac8288731b1180cf6cb483d85948e5efc046ce/include/cutlass/gemm/kernel/gemm_universal.hpp
- https://github.com/solrex/cutlass/blob/30ac8288731b1180cf6cb483d85948e5efc046ce/include/cutlass/gemm/kernel/sm70_gemm_blockscaled_accum.hpp
- https://github.com/solrex/cutlass/blob/30ac8288731b1180cf6cb483d85948e5efc046ce/examples/85_ada_ampere_gemm_with_blockwise_scaling/85a_ada_fp8_gemm_with_groupwise_scaling_cute.cu
evidence_api: https://api.github.com/repos/NVIDIA/cutlass/pulls/2328
---

# Add SM80/89 blockwise scaling kernel, support FP8 block/groupwise on Ada, INT8 on Ampere

[上游 PR](https://github.com/NVIDIA/cutlass/pull/2328) · 核查于 2026-09-17 · **open**

## 实现价值

提出 Ampere INT8 与 Ada FP8 的 block/groupwise scaling CuTe 路径；正文包含 A100、A10 和 4090。

## 使用边界

仍是 open PR；FP8 对应 SM89，INT8 对应 Ampere。不要把同页多架构标签理解成 Ampere 支持原生 FP8。

本地 GPU 验证：未运行。

## 固定版本源码

PR head：`30ac8288731b1180cf6cb483d85948e5efc046ce`。共 11 个 changed files；以下是实现阅读入口，不是完整变更清单。

- [include/cutlass/gemm/collective/collective_mma.hpp](https://github.com/solrex/cutlass/blob/30ac8288731b1180cf6cb483d85948e5efc046ce/include/cutlass/gemm/collective/collective_mma.hpp)
- [include/cutlass/gemm/collective/sm80_mma_multistage_blockwise_scaling.hpp](https://github.com/solrex/cutlass/blob/30ac8288731b1180cf6cb483d85948e5efc046ce/include/cutlass/gemm/collective/sm80_mma_multistage_blockwise_scaling.hpp)
- [include/cutlass/gemm/dispatch_policy.hpp](https://github.com/solrex/cutlass/blob/30ac8288731b1180cf6cb483d85948e5efc046ce/include/cutlass/gemm/dispatch_policy.hpp)
- [include/cutlass/gemm/kernel/gemm_universal.hpp](https://github.com/solrex/cutlass/blob/30ac8288731b1180cf6cb483d85948e5efc046ce/include/cutlass/gemm/kernel/gemm_universal.hpp)
- [include/cutlass/gemm/kernel/sm70_gemm_blockscaled_accum.hpp](https://github.com/solrex/cutlass/blob/30ac8288731b1180cf6cb483d85948e5efc046ce/include/cutlass/gemm/kernel/sm70_gemm_blockscaled_accum.hpp)
- [examples/85_ada_ampere_gemm_with_blockwise_scaling/85a_ada_fp8_gemm_with_groupwise_scaling_cute.cu](https://github.com/solrex/cutlass/blob/30ac8288731b1180cf6cb483d85948e5efc046ce/examples/85_ada_ampere_gemm_with_blockwise_scaling/85a_ada_fp8_gemm_with_groupwise_scaling_cute.cu)

## 讨论入口

- [讨论 3055765401](https://github.com/NVIDIA/cutlass/pull/2328#issuecomment-3055765401)（2025-07-10）
- [讨论 3056075279](https://github.com/NVIDIA/cutlass/pull/2328#issuecomment-3056075279)（2025-07-10）
- [讨论 3138792860](https://github.com/NVIDIA/cutlass/pull/2328#issuecomment-3138792860)（2025-07-31）
