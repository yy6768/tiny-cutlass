---
id: pr-cutlass-1604
title: Add Ampere GEMM example using Cute and CUTLASS 3.x
type: pr
url: https://github.com/NVIDIA/cutlass/pull/1604
repo: NVIDIA/cutlass
number: 1604
architectures:
- sm80
architecture_evidence:
  sm80: 提供 Ampere CuTe + CUTLASS 3.x Collective MMA/Epilogue 示例候选。 这是未合并 PR；不能宣称当前 CUTLASS 的所有 CollectiveBuilder
    均支持 SM80。
tags:
- cute
- cutlass3
- pipeline
kernel_types:
- gemm
symptoms:
- implementation-choice
status: open
confidence: source-reported
retrieved_at: '2026-09-17'
updated_at: '2024-11-12T17:04:56Z'
inclusion_reason: 提供 Ampere CuTe + CUTLASS 3.x Collective MMA/Epilogue 示例候选。
local_validation: not-run
head_sha: 205e4f3942738142eb7982884c3258066dc9741d
merged_at: null
merge_commit_sha: 9abf511ac290833fc251d29e5bcf010cc59c25e6
changed_files: 2
code_urls:
- https://github.com/intel/sycl-tla/blob/205e4f3942738142eb7982884c3258066dc9741d/examples/14_ampere_tf32_tensorop_gemm/ampere_tf32_tensorop_gemm_cute.cu
evidence_api: https://api.github.com/repos/NVIDIA/cutlass/pulls/1604
---

# Add Ampere GEMM example using Cute and CUTLASS 3.x

[上游 PR](https://github.com/NVIDIA/cutlass/pull/1604) · 核查于 2026-09-17 · **open**

## 实现价值

提供 Ampere CuTe + CUTLASS 3.x Collective MMA/Epilogue 示例候选。

## 使用边界

这是未合并 PR；不能宣称当前 CUTLASS 的所有 CollectiveBuilder 均支持 SM80。

本地 GPU 验证：未运行。

## 固定版本源码

PR head：`205e4f3942738142eb7982884c3258066dc9741d`。共 2 个 changed files；以下是实现阅读入口，不是完整变更清单。

- [examples/14_ampere_tf32_tensorop_gemm/ampere_tf32_tensorop_gemm_cute.cu](https://github.com/intel/sycl-tla/blob/205e4f3942738142eb7982884c3258066dc9741d/examples/14_ampere_tf32_tensorop_gemm/ampere_tf32_tensorop_gemm_cute.cu)

## 讨论入口

- [讨论 2222542582](https://github.com/NVIDIA/cutlass/pull/1604#issuecomment-2222542582)（2024-07-11）
- [讨论 2228673896](https://github.com/NVIDIA/cutlass/pull/1604#issuecomment-2228673896)（2024-07-15）
- [讨论 2228984572](https://github.com/NVIDIA/cutlass/pull/1604#issuecomment-2228984572)（2024-07-15）
