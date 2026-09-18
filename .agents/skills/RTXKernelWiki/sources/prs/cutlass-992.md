---
id: pr-cutlass-992
title: Update fMHA kernels
type: pr
url: https://github.com/NVIDIA/cutlass/pull/992
repo: NVIDIA/cutlass
number: 992
architectures:
- sm89
architecture_evidence:
  sm89: 变更落在 kernel_forward.h、custom_mma_multistage.h、mma_from_smem.h 及 iterator；可用来追踪
    2.x attention 的结构演进。
tags:
- cutlass-2x
- online-softmax
- mma-from-smem
- register-pressure
kernel_types:
- attention
symptoms:
- synchronization
- register-pressure
status: merged
confidence: inferred
retrieved_at: '2026-09-18'
updated_at: '2023-07-13T02:30:47Z'
inclusion_reason: example 41 更新：迭代 softmax、同步、寄存器使用和从 shared memory 执行第二次 MMA。
local_validation: not-run
head_sha: 3e535f4065bd5b8f6a464096087312760d26c567
merged_at: '2023-07-13T02:30:47Z'
merge_commit_sha: 146d314057c5f193a70c2b36896e739c8c60aef4
changed_files: 16
code_urls:
- https://github.com/NVIDIA/cutlass/blob/3e535f4065bd5b8f6a464096087312760d26c567/examples/41_fused_multi_head_attention/kernel_forward.h
- https://github.com/NVIDIA/cutlass/blob/3e535f4065bd5b8f6a464096087312760d26c567/examples/41_fused_multi_head_attention/gemm/mma_from_smem.h
- https://github.com/NVIDIA/cutlass/blob/3e535f4065bd5b8f6a464096087312760d26c567/examples/41_fused_multi_head_attention/gemm/custom_mma_multistage.h
evidence_api: https://api.github.com/repos/NVIDIA/cutlass/pulls/992
research_focus: sm89-local-attention
candidate_role: attention-mainline
priority: P1
---

# Update fMHA kernels

[上游 PR](https://github.com/NVIDIA/cutlass/pull/992) · merged `2023-07-13T02:30:47Z` · 核查 2026-09-18

## 对本任务的价值

**P1 / attention-mainline**。example 41 更新：迭代 softmax、同步、寄存器使用和从 shared memory 执行第二次 MMA。

变更落在 kernel_forward.h、custom_mma_multistage.h、mma_from_smem.h 及 iterator；可用来追踪 2.x attention 的结构演进。

## 适用边界

上游测试与计时不代表 4070/4090、L=16 小窗口的结果；融合主线仍需按本地 bias、reflect、Dq/Dv 和完整 block 语义适配。不是 FP8 attention 实现。

收录代表值得研究，不代表采用。GPU parity/性能：`not-run`。

## 固定版本实现

Head：`3e535f4065bd5b8f6a464096087312760d26c567`。PR 共 16 个变更文件；已抓取 15 个关键源码文件。

- [examples/41_fused_multi_head_attention/kernel_forward.h](../../artifacts/github/NVIDIA--cutlass/pr-992/20260917T165045842476Z/key-files/head/examples/41_fused_multi_head_attention/kernel_forward.h) · [固定 SHA 上游](https://github.com/NVIDIA/cutlass/blob/3e535f4065bd5b8f6a464096087312760d26c567/examples/41_fused_multi_head_attention/kernel_forward.h)
- [examples/41_fused_multi_head_attention/gemm/mma_from_smem.h](../../artifacts/github/NVIDIA--cutlass/pr-992/20260917T165045842476Z/key-files/head/examples/41_fused_multi_head_attention/gemm/mma_from_smem.h) · [固定 SHA 上游](https://github.com/NVIDIA/cutlass/blob/3e535f4065bd5b8f6a464096087312760d26c567/examples/41_fused_multi_head_attention/gemm/mma_from_smem.h)
- [examples/41_fused_multi_head_attention/gemm/custom_mma_multistage.h](../../artifacts/github/NVIDIA--cutlass/pr-992/20260917T165045842476Z/key-files/head/examples/41_fused_multi_head_attention/gemm/custom_mma_multistage.h) · [固定 SHA 上游](https://github.com/NVIDIA/cutlass/blob/3e535f4065bd5b8f6a464096087312760d26c567/examples/41_fused_multi_head_attention/gemm/custom_mma_multistage.h)

[正文、讨论、完整 diff、来源清单和其余源码](../../wiki/candidates/pr-NVIDIA--cutlass-992.md)。自动 Wiki 保持未审核标记，人工筛选结论以本页为准。
