# Swin CUTLASS

这个目录保留一条 Swin 主路径：把 FasterTransformer 的 Swin
`WindowAttention` block 和 PatchEmbed 搬成可读、可验证的 CUTLASS 实现骨架。
外部 activation 契约固定为 NHWC：输入 `[B,H,W,C]`，输出 `[B,H,W,C]`；
PatchEmbed 的输入也按 texture-like NHWC 处理。

## 文件结构

- `swin_problem.h`: `SwinProblem`、`SwinTensors` 和 shape/element 计算函数。
- `PATCH_EMBED.md`: PatchEmbed 的独立设计说明。
- `swin.h`: 对外共享入口，汇总 problem、device、kernel policy 和辅助转换声明。
- `kernel/default_swin.h`: 模板 policy factory，入口是
  `kernel::DefaultSwin<ArchTag, Element, ...>::Kernel`。
- `kernel/default_patch_embed.h`: PatchEmbed 的模板 policy factory。
- `device/swin.h`: device facade，入口是 `device::Swin<Kernel>`。
- `device/patch_embed.h`: PatchEmbed device facade。
- `kernel/swin.h`: CUDA kernel launch wrapper。
- `threadblock/layout.h`: Swin layout、bias 和 PatchEmbed glue 的 threadblock 级实现。
- `warp/layout.h`: window/patch 坐标映射 helper。
- `swin.cu`: WindowAttention 主路径的 device facade 实例化和 stage 调度实现。
- `patch_embed.cu`: PatchEmbed 的 CUTLASS Conv2d 调度实现。
- `trt/swin_plugin.{h,cpp}`: TensorRT plugin binding。
- `../tests/swin/window.cpp`: WindowAttention 主路径测试入口。
- `../tests/swin/patch_embed.cu`: PatchEmbed 测试入口。
- `../tests/swin/cudnn_swin_reference.h`: cuDNN frontend SDPA reference。

## 主路径

```text
PatchEmbed
WindowPartition
QKV projection
WindowAttention
OutputProjection
WindowReverse
```

`WindowAttention` 是主 attention 接口。QK、softmax、PV 不作为独立公开接口暴露，
它们在 fused tiled-online attention core 里面完成。

## FasterTransformer 对应表

| tiny-cutlass | FasterTransformer Swin |
| --- | --- |
| `device::PatchEmbed` | `patchEmbed` 里的 Conv2d + bias LayerNorm |
| `threadblock::WindowPartition` | `invokeShiftPartition` |
| `QKV projection` | `WindowAttention<T>::forward` 里的 QKV GEMM |
| `WindowAttention` | `FusedMHARunnerFP16v2::run` / fused local attention |
| `OutputProjection` | `attention_output_weight.kernel` GEMM |
| `threadblock::WindowReverse` | `invokeReverseRoll` |
| `threadblock::PatchMerge` | stage transition 的 image merge gather |

## CUTLASS 分层

```text
device::Swin<Kernel>
  kernel::DefaultSwin<ArchTag, Element, ...>::Kernel
    kernel::swin_threadblock_kernel<Threadblock>
      threadblock::{WindowPartition, AddQkvBiasSplit, AddBias, WindowReverse, PatchMerge}
        warp::{window_mapping, patch_merge_input_token}
```

QKV projection 和 output projection 使用 `cutlass::gemm::device::Gemm`。
PatchEmbed 使用 `cutlass::conv::device::ImplicitGemmConvolution`：

```text
NHWC image
  -> threadblock::ImagePadChannels
NHWC padded image
  -> CUTLASS implicit GEMM Conv2d, kernel=4x4, stride=4
NHWC patch embedding
  -> threadblock::AddBiasLayerNorm
NHWC tokens [B, H/4, W/4, embed_dim]
```

RGB texture 输入的 `C=3` 不满足 TensorOp optimized NHWC conv 的通道对齐，所以当前路径
显式 pad 到 `input_channels_padded=8`，filter 后 5 个通道补零；数学结果仍等价于官方
PatchEmbed。
WindowAttention 复用当前工程里 FlashAttention 02 的 tiled online attention core，
一个 kernel 内完成 `QK -> bias/mask -> online softmax -> PV`，不落全局
attention probability 矩阵。

## 当前验证约束

- dtype: FP16 input/weight/output，FP32 accumulation/compute。
- layout: activation 输入/输出固定 `[B,H,W,C]`；window token 是内部矩阵视图
  `[B*nW,L,C]`。
- TensorRT plugin: activation input/output 只接受 `kHALF` + `kLINEAR`，
  并在 enqueue 检查实际 shape 为 `[B,H,W,C]`。
- `C = head_number * head_size`。
- `image_size % window_size == 0`。
- `0 <= shift_size < window_size`。
- `head_size % 8 == 0`，`C % 8 == 0`。
- `window_len <= 64`。
- 运行设备需要 Ampere 或更新架构。
- 构建测试需要 cuDNN 和 cudnn-frontend；`CUDNN_PATH` 或
  `TINY_CUTLASS_CUDNN_PATH` 必须可用。

## 构建和验证

```bat
scripts\kernels\swin\swin.bat
```

这个入口执行 build -> verify -> bench。设置 `CUTLASS_PROFILE=1` 时，
`bench.py` 会额外在 `build/reports/swin/` 生成 `.nsys-rep`、SQLite 和 CSV stats。
