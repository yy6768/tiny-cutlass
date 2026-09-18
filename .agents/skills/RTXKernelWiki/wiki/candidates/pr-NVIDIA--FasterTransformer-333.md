---
id: candidate-pr-NVIDIA--FasterTransformer-333
title: Fix/swin qk scale
type: pr
repo: NVIDIA/FasterTransformer
architectures:
- sm89
status: merged
tags: []
review: defer
url: https://github.com/NVIDIA/FasterTransformer/pull/333
inclusion_reason: 自动抓取 candidate；架构来自搜索命中，需阅读代码确认。
local_validation: not-run
retrieved_at: 20260917T165141113463Z
artifact_dir: artifacts/github/NVIDIA--FasterTransformer/pr-333/20260917T165141113463Z
---

# Fix/swin qk scale

候选状态：**defer / 未审核**。架构标签是发现线索，不是设备支持证明。

- [GitHub](https://github.com/NVIDIA/FasterTransformer/pull/333)
- [原始正文与讨论](../../artifacts/github/NVIDIA--FasterTransformer/pr-333/20260917T165141113463Z/page.md)
- [来源清单](../../artifacts/github/NVIDIA--FasterTransformer/pr-333/20260917T165141113463Z/PROVENANCE.yaml)
- [完整 PR diff](../../artifacts/github/NVIDIA--FasterTransformer/pr-333/20260917T165141113463Z/diff.patch)

Head SHA: `8fc725235bf3c707945e0ccd326783a1aafdff4b` · merged

## 关键实现文件

- [src/fastertransformer/layers/attention_layers/WindowAttention.cc](../../artifacts/github/NVIDIA--FasterTransformer/pr-333/20260917T165141113463Z/key-files/head/src/fastertransformer/layers/attention_layers/WindowAttention.cc)（modified，`8fc725235bf3`）
- [src/fastertransformer/models/swin/Swin.h](../../artifacts/github/NVIDIA--FasterTransformer/pr-333/20260917T165141113463Z/key-files/head/src/fastertransformer/models/swin/Swin.h)（modified，`8fc725235bf3`）
- [src/fastertransformer/models/swin_int8/SwinINT8.h](../../artifacts/github/NVIDIA--FasterTransformer/pr-333/20260917T165141113463Z/key-files/head/src/fastertransformer/models/swin_int8/SwinINT8.h)（modified，`8fc725235bf3`）
- [src/fastertransformer/utils/gemm_test/decoding_gemm_func.cc](../../artifacts/github/NVIDIA--FasterTransformer/pr-333/20260917T165141113463Z/key-files/head/src/fastertransformer/utils/gemm_test/decoding_gemm_func.cc)（modified，`8fc725235bf3`）
- [src/fastertransformer/utils/gemm_test/gpt_gemm_func.cc](../../artifacts/github/NVIDIA--FasterTransformer/pr-333/20260917T165141113463Z/key-files/head/src/fastertransformer/utils/gemm_test/gpt_gemm_func.cc)（modified，`8fc725235bf3`）
- [src/fastertransformer/utils/gemm_test/t5_gemm_func.cc](../../artifacts/github/NVIDIA--FasterTransformer/pr-333/20260917T165141113463Z/key-files/head/src/fastertransformer/utils/gemm_test/t5_gemm_func.cc)（modified，`8fc725235bf3`）
