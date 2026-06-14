# Swin PatchEmbed

这个文档单独记录 Swin PatchEmbed 的 CUTLASS 实现。目标是把官方 PyTorch
`PatchEmbed` 的 `Conv2d(kernel=patch_size, stride=patch_size) + flatten +
LayerNorm` 映射成一条可验证的 device 路径，同时把本工程的外部 activation
契约固定为 NHWC。

## 对齐的语义

官方 Swin v1 tiny 的 PyTorch 路径是：

```text
input image [B, C, H, W]
  -> Conv2d, out_channels=embed_dim, kernel=patch_size, stride=patch_size
conv output [B, K, H/patch, W/patch]
  -> flatten spatial dims
tokens [B, H/patch * W/patch, K]	
  -> LayerNorm over K
```

本工程的外部 activation 契约固定为 NHWC，接近 texture 输入：

```text
input  : [B, H, W, C]
output : [B, H/patch, W/patch, K]
```

其中 `K = embed_dim`。输出没有物理 flatten 成 `[B, L, K]`，但连续 NHWC
storage 等价于按 token 顺序访问 `[B, H/patch * W/patch, K]`。

PyTorch 权重仍按官方 `OIHW` 传入：

```text
kernel : [K, C, R, S]
bias   : [K]
gamma  : [K]
beta   : [K]
```

## 代码入口

主入口在：

- `swin_problem.h`: `PatchEmbedProblem`、`PatchEmbedTensors<Element>` 和元素数量函数。
- `kernel/default_patch_embed.h`: `DefaultPatchEmbed<ArchTag, Element, ...>` policy factory。
- `device/patch_embed.h`: `device::PatchEmbed<Kernel>` facade。
- `patch_embed.cu`: device 调度实现。
- `threadblock/layout.h`: channel pad、filter reorder/pad、BiasLayerNorm 的 threadblock stage。
- `../tests/swin/patch_embed.cu`: host reference、正确性和 benchmark harness。

调用形态是模板 policy 风格：

```cpp
using Kernel = typename kernel::DefaultPatchEmbed<
    ArchTag,
    Element>::Kernel;

using Op = device::PatchEmbed<Kernel>;
cutlass::Status status = Op::run(problem, tensors, stream);
```

具体架构、元素类型和 tile shape 只应该出现在显式实例化、测试或 launcher 层；
公共 API 不提供把 dtype、arch 或 layout 写进名字的 concrete alias。

## 数据流

当前 device path 是：

```text
NHWC input
  -> threadblock::ImagePadChannels
padded NHWC input
  -> threadblock::FilterOihwToKrscPadded
padded KRSC filter
  -> cutlass::conv::device::ImplicitGemmConvolution
NHWC conv output
  -> threadblock::AddBiasLayerNorm
NHWC PatchEmbed output
```

### 1. ImagePadChannels

RGB 输入 `C=3` 不能直接满足 TensorOp optimized NHWC conv 的通道对齐要求，所以先 pad
到 `input_channels_padded=8`：

```text
[B, H, W, 3] -> [B, H, W, 8]
```

真实通道拷贝，padding 通道写 0。这样数学结果仍等价于原始 `C=3` conv。

### 2. FilterOihwToKrscPadded

官方 PyTorch 权重是 `OIHW`：

```text
[K, C, R, S]
```

CUTLASS NHWC Conv2d 的 filter tensor 按 `KRSC` 视图传入，所以这里做一次 reorder，
并对输入通道维补 0：

```text
[K, C, R, S] -> [K, R, S, C_pad]
```

### 3. CUTLASS Conv2d

`DefaultPatchEmbed` 生成 CUTLASS Conv2d fprop kernel：

```text
cutlass::conv::kernel::DefaultConv2dFprop<
  Element, TensorNHWC,
  Element, TensorNHWC,
  Element, TensorNHWC,
  ElementAccumulator,
  OpClassTensorOp,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  GemmIdentityThreadblockSwizzle<1>,
  kStages,
  OpMultiplyAdd,
  IteratorAlgorithm::kOptimized,
  StrideSupport::kStrided
>
```

Conv problem 对应：

```text
activation : [B, H, W, C_pad]
filter     : [K, patch, patch, C_pad]
padding    : 0
stride     : patch
dilation   : 1
output     : [B, H/patch, W/patch, K]
```

### 4. AddBiasLayerNorm

Conv 输出先加 `bias[K]`，再对每个 patch token 的 `K` 维做 LayerNorm：

```text
y = LayerNorm(conv(x) + bias, gamma, beta, eps)
```

实现上每个 token 一个 threadblock，shared memory 放 `sum` 和 `square_sum`，最后写回
NHWC output。

## Workspace

`PatchEmbedTensors<Element>::conv_output` 是 workspace base，而不是只存 conv output。
布局固定为：

```text
[0, output_elements)
    conv output [B, H/patch, W/patch, K]

next input_padded_elements
    padded NHWC activation [B, H, W, C_pad]

next kernel_padded_elements
    padded KRSC filter [K, patch, patch, C_pad]
```

workspace element 数量：

```cpp
patch_embed_output_elements(problem)
  + patch_embed_input_padded_elements(problem)
  + patch_embed_kernel_padded_elements(problem)
```

`tensors.output` 指向最终 NHWC PatchEmbed output，可以和外部下游 buffer 分开。

## 支持约束

`device::PatchEmbed<Kernel>::can_implement` 会显式拒绝不支持的配置：

- `batch_size > 0`
- `image_size > 0`
- `in_channels > 0`
- `input_channels_padded >= in_channels`
- `embed_dim > 0`
- `patch_size > 0`
- `layernorm_eps > 0`
- `image_size % patch_size == 0`
- `input_channels_padded % 8 == 0`
- `embed_dim % 8 == 0`

当前显式实例化路径选择 TensorOp policy；不支持的 shape 或类型通过
`cutlass::Status` / `can_implement` 返回失败，不做 SIMT 或 cuDNN fallback。

## 验证和性能入口

单独运行 PatchEmbed harness：

```bat
build\csrc\swin\Release\swin_patch_embed.exe ^
  --batch_size=1 ^
  --image_size=224 ^
  --in_channels=3 ^
  --input_channels_padded=8 ^
  --embed_dim=96 ^
  --patch_size=4 ^
  --iterations=50
```

完整 Swin workflow：

```bat
scripts\kernels\swin\swin.bat
```

验证脚本会先跑 PatchEmbed host reference parity，再跑 Swin WindowAttention parity：

```bat
python scripts\kernels\swin\verify.py --build-dir build --config Release
```

benchmark 和 nsys：

```bat
python scripts\kernels\swin\bench.py ^
  --build-dir build ^
  --config Release ^
  --batch-size 1 ^
  --iterations 100 ^
  --patch-iterations 50 ^
  --nsys ^
  --nsys-iterations 3
```

报告写入：

```text
build/reports/swin/bench_b1.csv
build/reports/swin/bench_b1.json
build/reports/swin/patch_embed_b1.nsys-rep
build/reports/swin/patch_embed_b1_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv
build/reports/swin/patch_embed_b1_cuda_api_sum_cuda_api_sum.csv
```
