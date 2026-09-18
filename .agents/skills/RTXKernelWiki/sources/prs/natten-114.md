---
id: pr-natten-114
title: Disable 64x128x128 GEMM config for SM86 and 89
type: pr
url: https://github.com/SHI-Labs/NATTEN/pull/114
repo: SHI-Labs/NATTEN
number: 114
architectures:
- sm89
architecture_evidence:
  sm89: src/natten/autotuner.py 在 compute capability 86/89 时移除大 tile；作者明确只测过 SM80/SM86，SM89
    当时是保守限制。
tags:
- fna
- tiling
- shared-memory
kernel_types:
- attention
symptoms:
- launch-failure
- shared-memory-limit
status: merged
confidence: source-reported
retrieved_at: '2026-09-18'
updated_at: '2024-03-16T02:50:58Z'
inclusion_reason: 为 SM86/SM89 排除 64×128×128 GEMM 配置，避免照搬 A100 的 shared-memory 用量。
local_validation: not-run
head_sha: 12ed5ea7ed92c64ef1327a439382525f4e35b034
merged_at: '2024-03-16T02:50:54Z'
merge_commit_sha: e4ceaef3ad8817b3c02ba55ec0dcddd532eb8f5d
changed_files: 2
code_urls:
- https://github.com/SHI-Labs/NATTEN/blob/12ed5ea7ed92c64ef1327a439382525f4e35b034/src/natten/autotuner.py
- https://github.com/SHI-Labs/NATTEN/blob/12ed5ea7ed92c64ef1327a439382525f4e35b034/src/natten/functional.py
evidence_api: https://api.github.com/repos/SHI-Labs/NATTEN/pulls/114
research_focus: sm89-local-attention
candidate_role: resource-constraint
priority: P0
---

# Disable 64x128x128 GEMM config for SM86 and 89

[上游 PR](https://github.com/SHI-Labs/NATTEN/pull/114) · merged `2024-03-16T02:50:54Z` · 核查 2026-09-18

## 对本任务的价值

**P0 / resource-constraint**。为 SM86/SM89 排除 64×128×128 GEMM 配置，避免照搬 A100 的 shared-memory 用量。

src/natten/autotuner.py 在 compute capability 86/89 时移除大 tile；作者明确只测过 SM80/SM86，SM89 当时是保守限制。

## 适用边界

这是历史版本的 autotuner 策略，不能推成所有 head-dim、dtype 或后续版本都禁止某个 tile；仍需对具体模板实例核算 shared memory。不是 SM89 实测性能证据。

收录代表值得研究，不代表采用。GPU parity/性能：`not-run`。

## 固定版本实现

Head：`12ed5ea7ed92c64ef1327a439382525f4e35b034`。PR 共 2 个变更文件；已抓取 2 个关键源码文件。

- [src/natten/autotuner.py](../../artifacts/github/SHI-Labs--NATTEN/pr-114/20260917T164959670620Z/key-files/head/src/natten/autotuner.py) · [固定 SHA 上游](https://github.com/SHI-Labs/NATTEN/blob/12ed5ea7ed92c64ef1327a439382525f4e35b034/src/natten/autotuner.py)
- [src/natten/functional.py](../../artifacts/github/SHI-Labs--NATTEN/pr-114/20260917T164959670620Z/key-files/head/src/natten/functional.py) · [固定 SHA 上游](https://github.com/SHI-Labs/NATTEN/blob/12ed5ea7ed92c64ef1327a439382525f4e35b034/src/natten/functional.py)

[正文、讨论、完整 diff、来源清单和其余源码](../../wiki/candidates/pr-SHI-Labs--NATTEN-114.md)。自动 Wiki 保持未审核标记，人工筛选结论以本页为准。
