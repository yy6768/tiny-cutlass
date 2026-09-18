---
id: candidate-pr-thu-ml--SageAttention-196
title: Update to SageAttention v2.2.0 (sage2++)
type: pr
repo: thu-ml/SageAttention
architectures:
- sm89
status: merged
tags: []
review: defer
url: https://github.com/thu-ml/SageAttention/pull/196
inclusion_reason: 自动抓取 candidate；架构来自搜索命中，需阅读代码确认。
local_validation: not-run
retrieved_at: 20260917T165114654675Z
artifact_dir: artifacts/github/thu-ml--SageAttention/pr-196/20260917T165114654675Z
---

# Update to SageAttention v2.2.0 (sage2++)

候选状态：**defer / 未审核**。架构标签是发现线索，不是设备支持证明。

- [GitHub](https://github.com/thu-ml/SageAttention/pull/196)
- [原始正文与讨论](../../artifacts/github/thu-ml--SageAttention/pr-196/20260917T165114654675Z/page.md)
- [来源清单](../../artifacts/github/thu-ml--SageAttention/pr-196/20260917T165114654675Z/PROVENANCE.yaml)
- [完整 PR diff](../../artifacts/github/thu-ml--SageAttention/pr-196/20260917T165114654675Z/diff.patch)

Head SHA: `a6b3b19e13388a629e7b0c0f6647265a864cd6cf` · merged

## 关键实现文件

- [csrc/mma.cuh](../../artifacts/github/thu-ml--SageAttention/pr-196/20260917T165114654675Z/key-files/head/csrc/mma.cuh)（modified，`a6b3b19e1338`）
- [csrc/numeric_conversion.cuh](../../artifacts/github/thu-ml--SageAttention/pr-196/20260917T165114654675Z/key-files/head/csrc/numeric_conversion.cuh)（modified，`a6b3b19e1338`）
- [csrc/qattn/attn_cuda_sm89.h](../../artifacts/github/thu-ml--SageAttention/pr-196/20260917T165114654675Z/key-files/head/csrc/qattn/attn_cuda_sm89.h)（modified，`a6b3b19e1338`）
- [csrc/qattn/attn_utils.cuh](../../artifacts/github/thu-ml--SageAttention/pr-196/20260917T165114654675Z/key-files/head/csrc/qattn/attn_utils.cuh)（modified，`a6b3b19e1338`）
- [csrc/qattn/pybind_sm89.cpp](../../artifacts/github/thu-ml--SageAttention/pr-196/20260917T165114654675Z/key-files/head/csrc/qattn/pybind_sm89.cpp)（modified，`a6b3b19e1338`）
- [csrc/qattn/qk_int_sv_f8_cuda_sm89.cu](../../artifacts/github/thu-ml--SageAttention/pr-196/20260917T165114654675Z/key-files/base/csrc/qattn/qk_int_sv_f8_cuda_sm89.cu)（removed，`e9b072f0fc26`）
- [csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh](../../artifacts/github/thu-ml--SageAttention/pr-196/20260917T165114654675Z/key-files/head/csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh)（added，`a6b3b19e1338`）
- [csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_attn_inst_buf.cu](../../artifacts/github/thu-ml--SageAttention/pr-196/20260917T165114654675Z/key-files/head/csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_attn_inst_buf.cu)（added，`a6b3b19e1338`）
- [csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu](../../artifacts/github/thu-ml--SageAttention/pr-196/20260917T165114654675Z/key-files/head/csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu)（added，`a6b3b19e1338`）
- [csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_attn.cu](../../artifacts/github/thu-ml--SageAttention/pr-196/20260917T165114654675Z/key-files/head/csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_attn.cu)（added，`a6b3b19e1338`）
- [csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_attn_inst_buf.cu](../../artifacts/github/thu-ml--SageAttention/pr-196/20260917T165114654675Z/key-files/head/csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_attn_inst_buf.cu)（added，`a6b3b19e1338`）
- [csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn.cu](../../artifacts/github/thu-ml--SageAttention/pr-196/20260917T165114654675Z/key-files/head/csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn.cu)（added，`a6b3b19e1338`）
- [csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf.cu](../../artifacts/github/thu-ml--SageAttention/pr-196/20260917T165114654675Z/key-files/head/csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf.cu)（added，`a6b3b19e1338`）
- [csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn.cu](../../artifacts/github/thu-ml--SageAttention/pr-196/20260917T165114654675Z/key-files/head/csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn.cu)（added，`a6b3b19e1338`）
- [sageattention/core.py](../../artifacts/github/thu-ml--SageAttention/pr-196/20260917T165114654675Z/key-files/head/sageattention/core.py)（modified，`a6b3b19e1338`）
