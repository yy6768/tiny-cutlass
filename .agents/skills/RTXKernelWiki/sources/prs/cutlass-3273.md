---
id: pr-cutlass-3273
title: '[CuTeDSL] Add SM120 MXF4/NVFP4 native-TMA path'
type: pr
url: https://github.com/NVIDIA/cutlass/pull/3273
repo: NVIDIA/cutlass
number: 3273
architectures:
- sm120
architecture_evidence:
  sm120: 提出 SM120 warp MMA + native TMA 的 CuTe DSL 实现，覆盖 scale fragment、copy layout 和 microtile 示例。 closed-unmerged
    候选，不证明已进入上游；使用固定 head 阅读，并优先对照当前官方 Blackwell GeForce 示例。
tags:
- nvfp4
- tma
- cute-dsl
- block-scaling
kernel_types:
- gemm
symptoms:
- implementation-choice
status: closed-unmerged
confidence: source-reported
retrieved_at: '2026-09-17'
updated_at: '2026-07-20T14:37:04Z'
inclusion_reason: 提出 SM120 warp MMA + native TMA 的 CuTe DSL 实现，覆盖 scale fragment、copy layout 和 microtile 示例。
local_validation: not-run
head_sha: 750b98567e3b8fc3e1e502242235b669e8527f34
merged_at: null
merge_commit_sha: null
changed_files: 19
code_urls:
- https://github.com/alecco/cutlass/blob/750b98567e3b8fc3e1e502242235b669e8527f34/examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/sm120_mxf4nvf4_native_tma_microtile.py
- https://github.com/alecco/cutlass/blob/750b98567e3b8fc3e1e502242235b669e8527f34/python/CuTeDSL/cutlass/cute/__init__.py
- https://github.com/alecco/cutlass/blob/750b98567e3b8fc3e1e502242235b669e8527f34/python/CuTeDSL/cutlass/cute/algorithm.py
- https://github.com/alecco/cutlass/blob/750b98567e3b8fc3e1e502242235b669e8527f34/python/CuTeDSL/cutlass/cute/nvgpu/warp/__init__.py
- https://github.com/alecco/cutlass/blob/750b98567e3b8fc3e1e502242235b669e8527f34/python/CuTeDSL/cutlass/cute/nvgpu/warp/mma.py
- https://github.com/alecco/cutlass/blob/750b98567e3b8fc3e1e502242235b669e8527f34/python/CuTeDSL/cutlass/cute/tensor.py
evidence_api: https://api.github.com/repos/NVIDIA/cutlass/pulls/3273
---

# [CuTeDSL] Add SM120 MXF4/NVFP4 native-TMA path

[上游 PR](https://github.com/NVIDIA/cutlass/pull/3273) · 核查于 2026-09-17 · **closed-unmerged**

## 实现价值

提出 SM120 warp MMA + native TMA 的 CuTe DSL 实现，覆盖 scale fragment、copy layout 和 microtile 示例。

## 使用边界

closed-unmerged 候选，不证明已进入上游；使用固定 head 阅读，并优先对照当前官方 Blackwell GeForce 示例。

本地 GPU 验证：未运行。

## 固定版本源码

PR head：`750b98567e3b8fc3e1e502242235b669e8527f34`。共 19 个 changed files；以下是实现阅读入口，不是完整变更清单。

- [examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/sm120_mxf4nvf4_native_tma_microtile.py](https://github.com/alecco/cutlass/blob/750b98567e3b8fc3e1e502242235b669e8527f34/examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/sm120_mxf4nvf4_native_tma_microtile.py)
- [python/CuTeDSL/cutlass/cute/__init__.py](https://github.com/alecco/cutlass/blob/750b98567e3b8fc3e1e502242235b669e8527f34/python/CuTeDSL/cutlass/cute/__init__.py)
- [python/CuTeDSL/cutlass/cute/algorithm.py](https://github.com/alecco/cutlass/blob/750b98567e3b8fc3e1e502242235b669e8527f34/python/CuTeDSL/cutlass/cute/algorithm.py)
- [python/CuTeDSL/cutlass/cute/nvgpu/warp/__init__.py](https://github.com/alecco/cutlass/blob/750b98567e3b8fc3e1e502242235b669e8527f34/python/CuTeDSL/cutlass/cute/nvgpu/warp/__init__.py)
- [python/CuTeDSL/cutlass/cute/nvgpu/warp/mma.py](https://github.com/alecco/cutlass/blob/750b98567e3b8fc3e1e502242235b669e8527f34/python/CuTeDSL/cutlass/cute/nvgpu/warp/mma.py)
- [python/CuTeDSL/cutlass/cute/tensor.py](https://github.com/alecco/cutlass/blob/750b98567e3b8fc3e1e502242235b669e8527f34/python/CuTeDSL/cutlass/cute/tensor.py)
