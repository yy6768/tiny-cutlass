---
id: doc-cutlass-implementation-map
title: CUTLASS 固定版本实现入口
type: doc
url: https://github.com/NVIDIA/cutlass/tree/147295a3d4b75f3aeff247c25b8927cea9a7006a
architectures:
- sm80
- sm86
- sm89
- sm120
architecture_evidence:
  sm80: 路径经 GitHub tree API 核实存在；架构由官方例子与代码选择确定，可移植设备尚未本地验证。
  sm86: 路径经 GitHub tree API 核实存在；架构由官方例子与代码选择确定，可移植设备尚未本地验证。
  sm89: 路径经 GitHub tree API 核实存在；架构由官方例子与代码选择确定，可移植设备尚未本地验证。
  sm120: 路径经 GitHub tree API 核实存在；架构由官方例子与代码选择确定，可移植设备尚未本地验证。
tags:
- cutlass2
- cutlass3
- cute
- cute-dsl
- mma-sync
- epilogue
kernel_types:
- gemm
- attention
- convolution
- grouped-gemm
symptoms:
- implementation-choice
status: published
confidence: inferred
retrieved_at: '2026-09-17'
inclusion_reason: 按实际算子找官方 main 中的实现；SM80 policy 的迁移、Ada FP8 和 SM120 GeForce 路线分别阅读。
local_validation: not-run
code_urls:
- https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/cute/tutorial/sgemm_sm80.cu
- https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/13_two_tensor_op_fusion/fused_two_convs_f16_sm80_shmem.cu
- https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/41_fused_multi_head_attention/kernel_forward.h
- https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/47_ampere_gemm_universal_streamk/ampere_gemm_universal_streamk_broadcast.cu
- https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/58_ada_fp8_gemm/ada_fp8_gemm.cu
- https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu
- https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu
- https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py
- https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent_cooperative.py
source_sha: 147295a3d4b75f3aeff247c25b8927cea9a7006a
---

# CUTLASS 固定版本实现入口

[原始来源](https://github.com/NVIDIA/cutlass/tree/147295a3d4b75f3aeff247c25b8927cea9a7006a) · 核查于 2026-09-17

## 采用的内容

按实际算子找官方 main 中的实现；SM80 policy 的迁移、Ada FP8 和 SM120 GeForce 路线分别阅读。

## 适用边界

这些是源码导航，不是本地编译结果。跨 SM80/86/89 标签参考 ArchTag 讨论，tile/stages 仍需调整。当前 main SHA 不代表用户 vendored CUTLASS 版本。

架构依据：路径经 GitHub tree API 核实存在；架构由官方例子与代码选择确定，可移植设备尚未本地验证。

本地 GPU 验证：未运行。

## 代码 / 实现入口

- [源码入口 1](https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/cute/tutorial/sgemm_sm80.cu)
- [源码入口 2](https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/13_two_tensor_op_fusion/fused_two_convs_f16_sm80_shmem.cu)
- [源码入口 3](https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/41_fused_multi_head_attention/kernel_forward.h)
- [源码入口 4](https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/47_ampere_gemm_universal_streamk/ampere_gemm_universal_streamk_broadcast.cu)
- [源码入口 5](https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/58_ada_fp8_gemm/ada_fp8_gemm.cu)
- [源码入口 6](https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu)
- [源码入口 7](https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu)
- [源码入口 8](https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py)
- [源码入口 9](https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent_cooperative.py)

## 按目标阅读

- SM80 / SM86 / SM89 GEMM：[源码](https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/cute/tutorial/sgemm_sm80.cu)
- SM80 / SM86 / SM89 back-to-back convolution：[源码](https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/13_two_tensor_op_fusion/fused_two_convs_f16_sm80_shmem.cu)
- SM80 / SM86 / SM89 attention：[源码](https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/41_fused_multi_head_attention/kernel_forward.h)
- SM80 / SM89 EVT：[源码](https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/47_ampere_gemm_universal_streamk/ampere_gemm_universal_streamk_broadcast.cu)
- SM89 FP8：[源码](https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/58_ada_fp8_gemm/ada_fp8_gemm.cu)
- SM120 NVFP4：[源码](https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu)
- SM120 grouped NVFP4：[源码](https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu)
- SM120 CuTe DSL dense GEMM：[源码](https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py)
- SM120 CuTe DSL blockscaled GEMM：[源码](https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent_cooperative.py)
