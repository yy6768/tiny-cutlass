---
id: pr-fastertransformer-333
title: Fix/swin qk scale
type: pr
url: https://github.com/NVIDIA/FasterTransformer/pull/333
repo: NVIDIA/FasterTransformer
number: 333
architectures:
- sm89
architecture_evidence:
  sm89: WindowAttention.cc 为 FusedMHARunnerFP16v2 重新计算 scale；window 相关性来自源码，对 SM89
    仅作语义参考。
tags:
- window-attention
- scaling
- swin
kernel_types:
- attention
symptoms:
- incorrect-output
- scale-semantics
status: merged
confidence: inferred
retrieved_at: '2026-09-18'
updated_at: '2022-09-29T05:16:14Z'
inclusion_reason: Swin WindowAttention 的 qk_scale 修正，用来检查接口 scale 和 fused runner 内部
  scale 是否重复。
local_validation: not-run
head_sha: 8fc725235bf3c707945e0ccd326783a1aafdff4b
merged_at: '2022-09-29T05:16:02Z'
merge_commit_sha: 7118182f19681d04ac26aa92250dedc8877ea456
changed_files: 6
code_urls:
- https://github.com/NVIDIA/FasterTransformer/blob/8fc725235bf3c707945e0ccd326783a1aafdff4b/src/fastertransformer/layers/attention_layers/WindowAttention.cc
evidence_api: https://api.github.com/repos/NVIDIA/FasterTransformer/pulls/333
research_focus: sm89-local-attention
candidate_role: window-semantics-reference
priority: P2
---

# Fix/swin qk scale

[上游 PR](https://github.com/NVIDIA/FasterTransformer/pull/333) · merged `2022-09-29T05:16:02Z` · 核查 2026-09-18

## 对本任务的价值

**P2 / window-semantics-reference**。Swin WindowAttention 的 qk_scale 修正，用来检查接口 scale 和 fused runner 内部 scale 是否重复。

WindowAttention.cc 为 FusedMHARunnerFP16v2 重新计算 scale；window 相关性来自源码，对 SM89 仅作语义参考。

## 适用边界

这是 FP16/TRT fused MHA 路线，不是 SM89 FP8 CUTLASS 2.x kernel。标准 Swin 的 mask/shift 不能替代本地 reflect/RMSNorm/window4 约定。

收录代表值得研究，不代表采用。GPU parity/性能：`not-run`。

## 固定版本实现

Head：`8fc725235bf3c707945e0ccd326783a1aafdff4b`。PR 共 6 个变更文件；已抓取 6 个关键源码文件。

- [src/fastertransformer/layers/attention_layers/WindowAttention.cc](../../artifacts/github/NVIDIA--FasterTransformer/pr-333/20260917T165141113463Z/key-files/head/src/fastertransformer/layers/attention_layers/WindowAttention.cc) · [固定 SHA 上游](https://github.com/NVIDIA/FasterTransformer/blob/8fc725235bf3c707945e0ccd326783a1aafdff4b/src/fastertransformer/layers/attention_layers/WindowAttention.cc)

[正文、讨论、完整 diff、来源清单和其余源码](../../wiki/candidates/pr-NVIDIA--FasterTransformer-333.md)。自动 Wiki 保持未审核标记，人工筛选结论以本页为准。
