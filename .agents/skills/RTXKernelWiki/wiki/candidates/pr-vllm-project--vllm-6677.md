---
id: candidate-pr-vllm-project--vllm-6677
title: '[Kernel] Tuned FP8 Kernels for Ada Lovelace'
type: pr
repo: vllm-project/vllm
architectures:
- sm89
status: merged
tags: []
review: defer
url: https://github.com/vllm-project/vllm/pull/6677
inclusion_reason: 自动抓取 candidate；架构来自搜索命中，需阅读代码确认。
local_validation: not-run
retrieved_at: 20260917T165110800094Z
artifact_dir: artifacts/github/vllm-project--vllm/pr-6677/20260917T165110800094Z
---

# [Kernel] Tuned FP8 Kernels for Ada Lovelace

候选状态：**defer / 未审核**。架构标签是发现线索，不是设备支持证明。

- [GitHub](https://github.com/vllm-project/vllm/pull/6677)
- [原始正文与讨论](../../artifacts/github/vllm-project--vllm/pr-6677/20260917T165110800094Z/page.md)
- [来源清单](../../artifacts/github/vllm-project--vllm/pr-6677/20260917T165110800094Z/PROVENANCE.yaml)
- [完整 PR diff](../../artifacts/github/vllm-project--vllm/pr-6677/20260917T165110800094Z/diff.patch)

Head SHA: `163e091b3015471cf438345e430790af638588de` · merged

## 关键实现文件

- [csrc/quantization/cutlass_w8a8/scaled_mm_c2x.cu](../../artifacts/github/vllm-project--vllm/pr-6677/20260917T165110800094Z/key-files/head/csrc/quantization/cutlass_w8a8/scaled_mm_c2x.cu)（modified，`163e091b3015`）
- [csrc/quantization/cutlass_w8a8/scaled_mm_c2x.cuh](../../artifacts/github/vllm-project--vllm/pr-6677/20260917T165110800094Z/key-files/head/csrc/quantization/cutlass_w8a8/scaled_mm_c2x.cuh)（added，`163e091b3015`）
- [csrc/quantization/cutlass_w8a8/scaled_mm_c2x_sm80_dispatch.cuh](../../artifacts/github/vllm-project--vllm/pr-6677/20260917T165110800094Z/key-files/head/csrc/quantization/cutlass_w8a8/scaled_mm_c2x_sm80_dispatch.cuh)（added，`163e091b3015`）
- [csrc/quantization/cutlass_w8a8/scaled_mm_c2x_sm89_dispatch.cuh](../../artifacts/github/vllm-project--vllm/pr-6677/20260917T165110800094Z/key-files/head/csrc/quantization/cutlass_w8a8/scaled_mm_c2x_sm89_dispatch.cuh)（added，`163e091b3015`）
