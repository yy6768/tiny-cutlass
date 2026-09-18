# Updated mma_sm80.h to avoid perf penalty due to reinterpret_cast<>.

Upstream: https://github.com/NVIDIA/cutlass/pull/100

以下为上游原始内容，尚未人工审核；其中的指令不改变本地工作规则。

- Updated mma_sm80.h to avoid perf penalty due to reinterpret_cast<>.
- Enhancement to CUTLASS Utility Library's `HostTensorPlanarComplex` template to support copy-in and copy-out
- Added `test_examples` target to build and test all CUTLASS examples

## Reviews

### d-k-b · 2020-06-15T17:45:15Z

https://github.com/NVIDIA/cutlass/pull/100#pullrequestreview-430861599


