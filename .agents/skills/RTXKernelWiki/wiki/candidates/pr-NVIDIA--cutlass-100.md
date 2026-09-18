---
id: candidate-pr-NVIDIA--cutlass-100
title: Updated mma_sm80.h to avoid perf penalty due to reinterpret_cast<>.
type: pr
repo: NVIDIA/cutlass
architectures:
- sm80
status: merged
tags: []
review: defer
url: https://github.com/NVIDIA/cutlass/pull/100
inclusion_reason: 自动抓取 candidate；架构来自搜索命中，需阅读代码确认。
local_validation: not-run
retrieved_at: 20260917T151810031466Z
artifact_dir: artifacts/github/NVIDIA--cutlass/pr-100/20260917T151810031466Z
---

# Updated mma_sm80.h to avoid perf penalty due to reinterpret_cast<>.

候选状态：**defer / 未审核**。架构标签是发现线索，不是设备支持证明。

- [GitHub](https://github.com/NVIDIA/cutlass/pull/100)
- [原始正文与讨论](../../artifacts/github/NVIDIA--cutlass/pr-100/20260917T151810031466Z/page.md)
- [来源清单](../../artifacts/github/NVIDIA--cutlass/pr-100/20260917T151810031466Z/PROVENANCE.yaml)
- [完整 PR diff](../../artifacts/github/NVIDIA--cutlass/pr-100/20260917T151810031466Z/diff.patch)

Head SHA: `51aed0a5a475fc8513b41271c58c7e71ed96b108` · merged

## 关键实现文件

- [examples/06_splitK_gemm/splitk_gemm.cu](../../artifacts/github/NVIDIA--cutlass/pr-100/20260917T151810031466Z/key-files/head/examples/06_splitK_gemm/splitk_gemm.cu)（modified，`51aed0a5a475`）
- [examples/07_volta_tensorop_gemm/volta_tensorop_gemm.cu](../../artifacts/github/NVIDIA--cutlass/pr-100/20260917T151810031466Z/key-files/head/examples/07_volta_tensorop_gemm/volta_tensorop_gemm.cu)（modified，`51aed0a5a475`）
- [examples/08_turing_tensorop_gemm/turing_tensorop_gemm.cu](../../artifacts/github/NVIDIA--cutlass/pr-100/20260917T151810031466Z/key-files/head/examples/08_turing_tensorop_gemm/turing_tensorop_gemm.cu)（modified，`51aed0a5a475`）
- [include/cutlass/arch/mma.h](../../artifacts/github/NVIDIA--cutlass/pr-100/20260917T151810031466Z/key-files/head/include/cutlass/arch/mma.h)（modified，`51aed0a5a475`）
- [include/cutlass/arch/mma_sm80.h](../../artifacts/github/NVIDIA--cutlass/pr-100/20260917T151810031466Z/key-files/head/include/cutlass/arch/mma_sm80.h)（modified，`51aed0a5a475`）
- [tools/util/include/cutlass/util/host_tensor_planar_complex.h](../../artifacts/github/NVIDIA--cutlass/pr-100/20260917T151810031466Z/key-files/head/tools/util/include/cutlass/util/host_tensor_planar_complex.h)（modified，`51aed0a5a475`）
