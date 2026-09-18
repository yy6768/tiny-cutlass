---
id: pr-flash-attention-2751
title: 'fix(cute): head-dim-aware SM80 forward tile sizes to fix SMEM overflow on sm_86/sm_89'
type: pr
url: https://github.com/Dao-AILab/flash-attention/pull/2751
repo: Dao-AILab/flash-attention
number: 2751
architectures:
- sm80
- sm86
- sm89
architecture_evidence:
  sm80: 为 SM80 系列 attention forward 按 head dimension 选 tile，处理 SM86/89 的 SMEM 溢出。 open PR；SM80 代码路径可覆盖多种 CC，但 A100
    的资源预算不能用于 3090/4090。
  sm86: 为 SM80 系列 attention forward 按 head dimension 选 tile，处理 SM86/89 的 SMEM 溢出。 open PR；SM80 代码路径可覆盖多种 CC，但 A100
    的资源预算不能用于 3090/4090。
  sm89: 为 SM80 系列 attention forward 按 head dimension 选 tile，处理 SM86/89 的 SMEM 溢出。 open PR；SM80 代码路径可覆盖多种 CC，但 A100
    的资源预算不能用于 3090/4090。
tags:
- attention
- tiling
- shared-memory
- cute-dsl
kernel_types:
- attention
symptoms:
- resource-limit
status: open
confidence: source-reported
retrieved_at: '2026-09-17'
updated_at: '2026-07-31T16:47:30Z'
inclusion_reason: 为 SM80 系列 attention forward 按 head dimension 选 tile，处理 SM86/89 的 SMEM 溢出。
local_validation: not-run
head_sha: acb0dc1372873c8aef1e92dd0a28f3735929c70c
merged_at: null
merge_commit_sha: null
changed_files: 3
code_urls:
- https://github.com/arbi-dev/flash-attention/blob/acb0dc1372873c8aef1e92dd0a28f3735929c70c/flash_attn/cute/flash_fwd.py
- https://github.com/arbi-dev/flash-attention/blob/acb0dc1372873c8aef1e92dd0a28f3735929c70c/flash_attn/cute/interface.py
- https://github.com/arbi-dev/flash-attention/blob/acb0dc1372873c8aef1e92dd0a28f3735929c70c/tests/cute/test_sm80_head_dim256_smem.py
evidence_api: https://api.github.com/repos/Dao-AILab/flash-attention/pulls/2751
---

# fix(cute): head-dim-aware SM80 forward tile sizes to fix SMEM overflow on sm_86/sm_89

[上游 PR](https://github.com/Dao-AILab/flash-attention/pull/2751) · 核查于 2026-09-17 · **open**

## 实现价值

为 SM80 系列 attention forward 按 head dimension 选 tile，处理 SM86/89 的 SMEM 溢出。

## 使用边界

open PR；SM80 代码路径可覆盖多种 CC，但 A100 的资源预算不能用于 3090/4090。

本地 GPU 验证：未运行。

## 固定版本源码

PR head：`acb0dc1372873c8aef1e92dd0a28f3735929c70c`。共 3 个 changed files；以下是实现阅读入口，不是完整变更清单。

- [flash_attn/cute/flash_fwd.py](https://github.com/arbi-dev/flash-attention/blob/acb0dc1372873c8aef1e92dd0a28f3735929c70c/flash_attn/cute/flash_fwd.py)
- [flash_attn/cute/interface.py](https://github.com/arbi-dev/flash-attention/blob/acb0dc1372873c8aef1e92dd0a28f3735929c70c/flash_attn/cute/interface.py)
- [tests/cute/test_sm80_head_dim256_smem.py](https://github.com/arbi-dev/flash-attention/blob/acb0dc1372873c8aef1e92dd0a28f3735929c70c/tests/cute/test_sm80_head_dim256_smem.py)
