---
id: candidate-pr-NVIDIA--cutlass-992
title: Update fMHA kernels
type: pr
repo: NVIDIA/cutlass
architectures:
- sm89
status: merged
tags: []
review: defer
url: https://github.com/NVIDIA/cutlass/pull/992
inclusion_reason: 自动抓取 candidate；架构来自搜索命中，需阅读代码确认。
local_validation: not-run
retrieved_at: 20260917T165045842476Z
artifact_dir: artifacts/github/NVIDIA--cutlass/pr-992/20260917T165045842476Z
---

# Update fMHA kernels

候选状态：**defer / 未审核**。架构标签是发现线索，不是设备支持证明。

- [GitHub](https://github.com/NVIDIA/cutlass/pull/992)
- [原始正文与讨论](../../artifacts/github/NVIDIA--cutlass/pr-992/20260917T165045842476Z/page.md)
- [来源清单](../../artifacts/github/NVIDIA--cutlass/pr-992/20260917T165045842476Z/PROVENANCE.yaml)
- [完整 PR diff](../../artifacts/github/NVIDIA--cutlass/pr-992/20260917T165045842476Z/diff.patch)

Head SHA: `3e535f4065bd5b8f6a464096087312760d26c567` · merged

## 关键实现文件

- [examples/41_fused_multi_head_attention/default_fmha_grouped.h](../../artifacts/github/NVIDIA--cutlass/pr-992/20260917T165045842476Z/key-files/head/examples/41_fused_multi_head_attention/default_fmha_grouped.h)（modified，`3e535f4065bd`）
- [examples/41_fused_multi_head_attention/fmha_grouped.h](../../artifacts/github/NVIDIA--cutlass/pr-992/20260917T165045842476Z/key-files/head/examples/41_fused_multi_head_attention/fmha_grouped.h)（modified，`3e535f4065bd`）
- [examples/41_fused_multi_head_attention/fused_multi_head_attention_backward.cu](../../artifacts/github/NVIDIA--cutlass/pr-992/20260917T165045842476Z/key-files/head/examples/41_fused_multi_head_attention/fused_multi_head_attention_backward.cu)（modified，`3e535f4065bd`）
- [examples/41_fused_multi_head_attention/fused_multihead_attention_fixed_seqlen.cu](../../artifacts/github/NVIDIA--cutlass/pr-992/20260917T165045842476Z/key-files/head/examples/41_fused_multi_head_attention/fused_multihead_attention_fixed_seqlen.cu)（modified，`3e535f4065bd`）
- [examples/41_fused_multi_head_attention/fused_multihead_attention_variable_seqlen.cu](../../artifacts/github/NVIDIA--cutlass/pr-992/20260917T165045842476Z/key-files/head/examples/41_fused_multi_head_attention/fused_multihead_attention_variable_seqlen.cu)（modified，`3e535f4065bd`）
- [examples/41_fused_multi_head_attention/gemm/custom_mma_multistage.h](../../artifacts/github/NVIDIA--cutlass/pr-992/20260917T165045842476Z/key-files/head/examples/41_fused_multi_head_attention/gemm/custom_mma_multistage.h)（modified，`3e535f4065bd`）
- [examples/41_fused_multi_head_attention/gemm/custom_mma_pipelined.h](../../artifacts/github/NVIDIA--cutlass/pr-992/20260917T165045842476Z/key-files/head/examples/41_fused_multi_head_attention/gemm/custom_mma_pipelined.h)（modified，`3e535f4065bd`）
- [examples/41_fused_multi_head_attention/gemm/mma_from_smem.h](../../artifacts/github/NVIDIA--cutlass/pr-992/20260917T165045842476Z/key-files/head/examples/41_fused_multi_head_attention/gemm/mma_from_smem.h)（modified，`3e535f4065bd`）
- [examples/41_fused_multi_head_attention/gemm_kernel_utils.h](../../artifacts/github/NVIDIA--cutlass/pr-992/20260917T165045842476Z/key-files/head/examples/41_fused_multi_head_attention/gemm_kernel_utils.h)（modified，`3e535f4065bd`）
- [examples/41_fused_multi_head_attention/iterators/default_warp_iterator_from_smem.h](../../artifacts/github/NVIDIA--cutlass/pr-992/20260917T165045842476Z/key-files/head/examples/41_fused_multi_head_attention/iterators/default_warp_iterator_from_smem.h)（added，`3e535f4065bd`）
- [examples/41_fused_multi_head_attention/iterators/transpose_warp_iterator.h](../../artifacts/github/NVIDIA--cutlass/pr-992/20260917T165045842476Z/key-files/head/examples/41_fused_multi_head_attention/iterators/transpose_warp_iterator.h)（modified，`3e535f4065bd`）
- [examples/41_fused_multi_head_attention/iterators/warp_iterator_from_smem.h](../../artifacts/github/NVIDIA--cutlass/pr-992/20260917T165045842476Z/key-files/head/examples/41_fused_multi_head_attention/iterators/warp_iterator_from_smem.h)（modified，`3e535f4065bd`）
- [examples/41_fused_multi_head_attention/kernel_backward.h](../../artifacts/github/NVIDIA--cutlass/pr-992/20260917T165045842476Z/key-files/head/examples/41_fused_multi_head_attention/kernel_backward.h)（modified，`3e535f4065bd`）
- [examples/41_fused_multi_head_attention/kernel_forward.h](../../artifacts/github/NVIDIA--cutlass/pr-992/20260917T165045842476Z/key-files/head/examples/41_fused_multi_head_attention/kernel_forward.h)（modified，`3e535f4065bd`）
- [examples/41_fused_multi_head_attention/transform/tile_smem_loader.h](../../artifacts/github/NVIDIA--cutlass/pr-992/20260917T165045842476Z/key-files/head/examples/41_fused_multi_head_attention/transform/tile_smem_loader.h)（modified，`3e535f4065bd`）
