---
id: pr-sageattention-196
title: Update to SageAttention v2.2.0 (sage2++)
type: pr
url: https://github.com/thu-ml/SageAttention/pull/196
repo: thu-ml/SageAttention
number: 196
architectures:
- sm89
architecture_evidence:
  sm89: core.py 将 SM89 路由到 qk_int8_pv_fp8_cuda；新增 sm89_qk_int8_sv_f8_accum_f16_* 和
    csrc/mma.cuh 的 FP8→FP16 MMA。
tags:
- fp8
- int8
- mixed-precision
- accumulation
kernel_types:
- attention
symptoms:
- numerical-accuracy
status: merged
confidence: source-reported
retrieved_at: '2026-09-18'
updated_at: '2025-07-08T12:01:10Z'
inclusion_reason: SM89 的 INT8 QK + FP8 PV，以及 FP16 局部累加/FP32 缓冲的混合精度参考。
local_validation: not-run
head_sha: a6b3b19e13388a629e7b0c0f6647265a864cd6cf
merged_at: '2025-07-01T06:09:30Z'
merge_commit_sha: 912f05da07731a93fdc0f7c2298005f7dcb42c0d
changed_files: 18
code_urls:
- https://github.com/thu-ml/SageAttention/blob/a6b3b19e13388a629e7b0c0f6647265a864cd6cf/sageattention/core.py
- https://github.com/thu-ml/SageAttention/blob/a6b3b19e13388a629e7b0c0f6647265a864cd6cf/csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh
- https://github.com/thu-ml/SageAttention/blob/a6b3b19e13388a629e7b0c0f6647265a864cd6cf/csrc/mma.cuh
- https://github.com/thu-ml/SageAttention/blob/a6b3b19e13388a629e7b0c0f6647265a864cd6cf/csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu
evidence_api: https://api.github.com/repos/thu-ml/SageAttention/pulls/196
research_focus: sm89-local-attention
candidate_role: mixed-precision-reference
priority: P1
---

# Update to SageAttention v2.2.0 (sage2++)

[上游 PR](https://github.com/thu-ml/SageAttention/pull/196) · merged `2025-07-01T06:09:30Z` · 核查 2026-09-18

## 对本任务的价值

**P1 / mixed-precision-reference**。SM89 的 INT8 QK + FP8 PV，以及 FP16 局部累加/FP32 缓冲的混合精度参考。

core.py 将 SM89 路由到 qk_int8_pv_fp8_cuda；新增 sm89_qk_int8_sv_f8_accum_f16_* 和 csrc/mma.cuh 的 FP8→FP16 MMA。

## 适用边界

这是独立 CUDA/inline PTX 实现，不属于本地 CUTLASS 2.x 主线，也不是全 FP8 QK/PV 或 2D 邻域/reflect 实现。该 head 的 FP16 accumulator primitive 以 CUDA >=12.8 和 __CUDA_ARCH__>=890 宏启用。量化位置、scale_max 和累加方式须成套理解，不直接拷贝 PTX。

收录代表值得研究，不代表采用。GPU parity/性能：`not-run`。

## 固定版本实现

Head：`a6b3b19e13388a629e7b0c0f6647265a864cd6cf`。PR 共 18 个变更文件；已抓取 15 个关键源码文件。

- [sageattention/core.py](../../artifacts/github/thu-ml--SageAttention/pr-196/20260917T165114654675Z/key-files/head/sageattention/core.py) · [固定 SHA 上游](https://github.com/thu-ml/SageAttention/blob/a6b3b19e13388a629e7b0c0f6647265a864cd6cf/sageattention/core.py)
- [csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh](../../artifacts/github/thu-ml--SageAttention/pr-196/20260917T165114654675Z/key-files/head/csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh) · [固定 SHA 上游](https://github.com/thu-ml/SageAttention/blob/a6b3b19e13388a629e7b0c0f6647265a864cd6cf/csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh)
- [csrc/mma.cuh](../../artifacts/github/thu-ml--SageAttention/pr-196/20260917T165114654675Z/key-files/head/csrc/mma.cuh) · [固定 SHA 上游](https://github.com/thu-ml/SageAttention/blob/a6b3b19e13388a629e7b0c0f6647265a864cd6cf/csrc/mma.cuh)
- [csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu](../../artifacts/github/thu-ml--SageAttention/pr-196/20260917T165114654675Z/key-files/head/csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu) · [固定 SHA 上游](https://github.com/thu-ml/SageAttention/blob/a6b3b19e13388a629e7b0c0f6647265a864cd6cf/csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu)

[正文、讨论、完整 diff、来源清单和其余源码](../../wiki/candidates/pr-thu-ml--SageAttention-196.md)。自动 Wiki 保持未审核标记，人工筛选结论以本页为准。
