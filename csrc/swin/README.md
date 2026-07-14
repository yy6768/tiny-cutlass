# Swin CUTLASS

这个 workspace 实现三个独立 public operator：

```cpp
device::PatchEmbed<ArchTag, Element>
device::SwinAttention<ArchTag, Element>
device::SwinBlock<ArchTag, Element>
```

调用方直接选择 arch 和 element type。CUTLASS factory 留在 `.cu` 实现侧，不作为
public device facade 的 `Policy` 模板实参。

## 目录

- `swin.h`: public aggregate header。
- `swin_problem.h`: 三个 operator 的 problem/tensor descriptor 和 shape helper。
- `device/`: public facade 与内部 launch helper。
- `kernel/`: `DefaultXxx` factory 和 CUDA kernel wrapper。
- `threadblock/`, `warp/`: glue stage 与 window 坐标映射。
- `swin.cu`: SwinAttention/SwinBlock 调度和显式实例化。
- `patch_embed.cu`: PatchEmbed 调度和显式实例化。
- `trt/`: 当前 SwinAttention TensorRT plugin。
- `docs/00-*.md` ... `docs/06-*.md`: 按执行层次编号的学习与设计记录。
- `../tests/swin/`: host/cuDNN reference executable。

## Descriptor

- `PatchEmbedProblem/PatchEmbedTensors`
- `SwinAttentionProblem/SwinAttentionTensors`
- `SwinBlockProblem/SwinBlockTensors`
- `WindowAttentionTensors`: attention 与 block 共享的内部 projection/attention buffers

SwinBlock descriptor 不再污染 attention-only API；PatchMerging 也不再通过 nullable
output pointer 挂在 attention operator 上。

## 当前实例

当前只显式实例化：

```cpp
ArchTag = cutlass::arch::Sm80
Element = cutlass::half_t
```

构建目标使用 `CMAKE_CUDA_ARCHITECTURES=89`，可在本机 SM89 GPU 上运行这个 Ampere
TensorOp policy。不支持的 arch/dtype/problem shape 返回 `cutlass::Status` 错误，不降级。

## 构建、验证与性能

```bat
scripts\kernels\swin\run.bat
```

入口固定执行 build -> verify -> bench。verify 覆盖三个 public operator，并可读取
`checkpoint/manifest.json` 描述的官方 artifact。设置 `CUTLASS_PROFILE=1` 后，bench
阶段同时采集 Nsys、NCU 和 NCU CSV，不使用独立 profiling 脚本。

更完整的路径说明见 [docs/00-overview.md](docs/00-overview.md) 和
[docs/06-workflow-and-profiling.md](docs/06-workflow-and-profiling.md)。
