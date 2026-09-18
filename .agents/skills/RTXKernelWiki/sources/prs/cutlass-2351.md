---
id: pr-cutlass-2351
title: Fix sgemm_sm80 example bug
type: pr
url: https://github.com/NVIDIA/cutlass/pull/2351
repo: NVIDIA/cutlass
number: 2351
architectures:
- sm80
- sm86
- sm89
architecture_evidence:
  sm80: 修正 sgemm_sm80 示例中 copy fragment 的 K 维与 MMA fragment 不匹配问题。 异常与 copy/MMA 分块对应关系有关；打印 layout 不能替代 sanitizer
    和 reference。SM86/89 为同一路径的迁移。
  sm86: 修正 sgemm_sm80 示例中 copy fragment 的 K 维与 MMA fragment 不匹配问题。 异常与 copy/MMA 分块对应关系有关；打印 layout 不能替代 sanitizer
    和 reference。SM86/89 为同一路径的迁移。
  sm89: 修正 sgemm_sm80 示例中 copy fragment 的 K 维与 MMA fragment 不匹配问题。 异常与 copy/MMA 分块对应关系有关；打印 layout 不能替代 sanitizer
    和 reference。SM86/89 为同一路径的迁移。
tags:
- cute
- ldmatrix
- pipeline
kernel_types:
- gemm
symptoms:
- illegal-memory-access
- correctness
status: merged
confidence: inferred
retrieved_at: '2026-09-17'
updated_at: '2025-07-31T02:12:33Z'
inclusion_reason: 修正 sgemm_sm80 示例中 copy fragment 的 K 维与 MMA fragment 不匹配问题。
local_validation: not-run
head_sha: b9b4b40ed264e8a78f0d9ca354a5d920fd844521
merged_at: '2025-07-31T02:12:33Z'
merge_commit_sha: da47886e34bfc3bd2038a1fdff5dd889dc74af08
changed_files: 1
code_urls:
- https://github.com/botbw/cutlass/blob/b9b4b40ed264e8a78f0d9ca354a5d920fd844521/examples/cute/tutorial/sgemm_sm80.cu
evidence_api: https://api.github.com/repos/NVIDIA/cutlass/pulls/2351
---

# Fix sgemm_sm80 example bug

[上游 PR](https://github.com/NVIDIA/cutlass/pull/2351) · 核查于 2026-09-17 · **merged**

## 实现价值

修正 sgemm_sm80 示例中 copy fragment 的 K 维与 MMA fragment 不匹配问题。

## 使用边界

异常与 copy/MMA 分块对应关系有关；打印 layout 不能替代 sanitizer 和 reference。SM86/89 为同一路径的迁移。

本地 GPU 验证：未运行。

## 固定版本源码

PR head：`b9b4b40ed264e8a78f0d9ca354a5d920fd844521`。共 1 个 changed files；以下是实现阅读入口，不是完整变更清单。

- [examples/cute/tutorial/sgemm_sm80.cu](https://github.com/botbw/cutlass/blob/b9b4b40ed264e8a78f0d9ca354a5d920fd844521/examples/cute/tutorial/sgemm_sm80.cu)

## 讨论入口

- [讨论 3135191043](https://github.com/NVIDIA/cutlass/pull/2351#issuecomment-3135191043)（2025-07-30）
