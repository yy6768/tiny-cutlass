---
id: pr-flashinfer-2798
title: Upgrade cutlass 4.2.1 -> 4.4.2
type: pr
url: https://github.com/flashinfer-ai/flashinfer/pull/2798
repo: flashinfer-ai/flashinfer
number: 2798
architectures:
- sm120
architecture_evidence:
  sm120: 升级 CUTLASS 依赖并调整 kernel 构建配置，关联 SM120/SM121 NVFP4 与 TMA descriptor 故障。 必须查看 submodule SHA 和目标 launch 路径；正文/标题的版本号不同，以固定差异和源码为准。
tags:
- nvfp4
- tma
- grouped-gemm
- build-guard
kernel_types:
- grouped-gemm
- moe
symptoms:
- misaligned-address
status: merged
confidence: source-reported
retrieved_at: '2026-09-17'
updated_at: '2026-03-19T18:16:00Z'
inclusion_reason: 升级 CUTLASS 依赖并调整 kernel 构建配置，关联 SM120/SM121 NVFP4 与 TMA descriptor 故障。
local_validation: not-run
head_sha: 6fdaa519667ed9435c995aaac2fb10a27980757c
merged_at: '2026-03-19T17:59:19Z'
merge_commit_sha: 9276e441898789e3ff21c397ca942103bf76cb12
changed_files: 6
code_urls:
- https://github.com/kahyunnam/flashinfer/blob/6fdaa519667ed9435c995aaac2fb10a27980757c/csrc/nv_internal/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm/collective/builders/sm90_gmma_builder_mixed_input.inl
- https://github.com/kahyunnam/flashinfer/blob/6fdaa519667ed9435c995aaac2fb10a27980757c/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h
- https://github.com/kahyunnam/flashinfer/blob/6fdaa519667ed9435c995aaac2fb10a27980757c/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/launchers/fpA_intB_launcher_sm90.inl
- https://github.com/kahyunnam/flashinfer/blob/6fdaa519667ed9435c995aaac2fb10a27980757c/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_tma_ws_mixed_input_launcher.inl
- https://github.com/kahyunnam/flashinfer/blob/6fdaa519667ed9435c995aaac2fb10a27980757c/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch.h
evidence_api: https://api.github.com/repos/flashinfer-ai/flashinfer/pulls/2798
---

# Upgrade cutlass 4.2.1 -> 4.4.2

[上游 PR](https://github.com/flashinfer-ai/flashinfer/pull/2798) · 核查于 2026-09-17 · **merged**

## 实现价值

升级 CUTLASS 依赖并调整 kernel 构建配置，关联 SM120/SM121 NVFP4 与 TMA descriptor 故障。

## 使用边界

必须查看 submodule SHA 和目标 launch 路径；正文/标题的版本号不同，以固定差异和源码为准。

本地 GPU 验证：未运行。

## 固定版本源码

PR head：`6fdaa519667ed9435c995aaac2fb10a27980757c`。共 6 个 changed files；以下是实现阅读入口，不是完整变更清单。

- [csrc/nv_internal/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm/collective/builders/sm90_gmma_builder_mixed_input.inl](https://github.com/kahyunnam/flashinfer/blob/6fdaa519667ed9435c995aaac2fb10a27980757c/csrc/nv_internal/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm/collective/builders/sm90_gmma_builder_mixed_input.inl)
- [csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h](https://github.com/kahyunnam/flashinfer/blob/6fdaa519667ed9435c995aaac2fb10a27980757c/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h)
- [csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/launchers/fpA_intB_launcher_sm90.inl](https://github.com/kahyunnam/flashinfer/blob/6fdaa519667ed9435c995aaac2fb10a27980757c/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/launchers/fpA_intB_launcher_sm90.inl)
- [csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_tma_ws_mixed_input_launcher.inl](https://github.com/kahyunnam/flashinfer/blob/6fdaa519667ed9435c995aaac2fb10a27980757c/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_tma_ws_mixed_input_launcher.inl)
- [csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch.h](https://github.com/kahyunnam/flashinfer/blob/6fdaa519667ed9435c995aaac2fb10a27980757c/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch.h)

## 讨论入口

- [讨论 4087955052](https://github.com/flashinfer-ai/flashinfer/pull/2798#issuecomment-4087955052)（2026-03-19）
- [讨论 4092130515](https://github.com/flashinfer-ai/flashinfer/pull/2798#issuecomment-4092130515)（2026-03-19）
- [讨论 4092172413](https://github.com/flashinfer-ai/flashinfer/pull/2798#issuecomment-4092172413)（2026-03-19）
