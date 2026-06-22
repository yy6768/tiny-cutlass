# Swin BasicLayer 与整网

这个文档记录 `BasicLayer`（一个 stage）和 `SwinTransformer`（整网）的三方实现，
以及当前 tiny-cutlass 还没有 driver 这一层的现状。

## BasicLayer（一个 stage）

### 对齐的语义

```text
input  : [B, L, C]              L = H * W
output : [B, L', C']            若有 downsample: L'=L/4, C'=2C；否则 L'=L, C'=C
```

一个 stage 内：depth 个 block 顺序执行，block 的 shift 交替 0 / window//2；
所有 block 结束后做一次 PatchMerging（最后一个 stage 除外）。

### 官方 PyTorch 实现

```python
class BasicLayer(nn.Module):
    def __init__(self, ..., depth, ..., downsample=None, ...):
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,   # 交替
                ...)
            for i in range(depth)])
        self.downsample = downsample(input_resolution, dim=dim, ...) if downsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x
```

要点：

- `shift_size = 0 if i%2==0 else window//2`：偶 block W-MSA，奇 block SW-MSA，
  成对出现。所以 `depth` 通常是偶数（Swin-T `[2,2,6,2]`）。
- `downsample`（PatchMerging）在所有 block 之后，用 **block 级的
  `input_resolution`**（merge 前的分辨率）构造。
- 每个 block 的 dim 不变，downsample 才把 dim 翻倍、分辨率减半。

### FasterTransformer 实现

`SwinTransformerBasicLayer<T>::forward`，additional_params
`{depth, num_head, do_patch_merge, sm}`：

```text
loop i in [0, depth):
    shift_size = (i % 2 == 0) ? 0 : (window_size / 2)
    block.forward(... shift_size ...)        # ping-pong block_output_ + (i%2)*size
if do_patch_merge:
    patchMerge(block_output_+((depth-1)%2)*size -> output_tensor)
else:
    # 最后的 block (i==depth-1) 直接写 output_tensor
```

- block 之间用 ping-pong buffer（`(i%2)`）。
- `do_patch_merge` 由整网传入：`layer_num_ > 1` 且非最后一层。
- `patchMerge` 见 [PATCH_MERGING.md](PATCH_MERGING.md)：`invokeMergeLayernorm` + Gemm。

## 整网 SwinTransformer

### 对齐的语义

```text
image  : [B, 3, H, W]   (NCHW)
output : [B, num_classes]
```

### 官方 PyTorch 实现

```python
def forward_features(self, x):
    x = self.patch_embed(x)              # [B, L, C]
    if self.ape: x = x + self.absolute_pos_embed
    x = self.pos_drop(x)
    for layer in self.layers:            # BasicLayer x num_layers
        x = layer(x)
    x = self.norm(x)                     # 最终 LayerNorm, [B, L, C]
    x = self.avgpool(x.transpose(1, 2))  # [B, C, 1]
    x = torch.flatten(x, 1)              # [B, C]
    return x

def forward(self, x):
    return self.head(self.forward_features(x))   # Linear C->num_classes
```

stage 串联（Swin-T `embed_dim=96, depths=[2,2,6,2], heads=[3,6,12,24],
window=7, img=224`）：

```text
patch_embed:           [B, 3, 224, 224] -> [B, 3136, 96]
layer0 dim 96  res 56  -> merge -> [B, 784, 192]
layer1 dim 192 res 28  -> merge -> [B, 196, 384]
layer2 dim 384 res 14  -> merge -> [B,  49, 768]
layer3 dim 768 res 7   (无 merge)  [B,  49, 768]
norm(768) + avgpool -> [B, 768]
head -> [B, num_classes]
```

- `downsample=PatchMerging if i_layer < num_layers-1 else None`：前 3 个 stage
  有 merge，最后一个没有。
- `num_features = embed_dim * 2^(num_layers-1) = 768`。
- 默认 `ape=False`，所以一般没有 absolute position embedding。

### FasterTransformer 实现

`SwinTransformer<T>::forward`：

```text
patchEmbed(conv2d + invokeAddBiasLayernorm)         # 见 PATCH_EMBED.md
loop basic layers:                                   # ping-pong basic_layer_output_
    do_patch_merge = (layer_num_ > 1) && (i != layer_num_-1)
    basic_layer.forward(...)
invokeGeneralLayerNorm(...)                          # 最终 norm
# avgpool 用 stridedBatchedGemm 和全 1 向量相乘:
stridedBatchedGemm(m=dim, n=1, k=final_len,
    A=basic_layer_output, B=avg_pool_ones_, C=output,
    batchCount=batch, alpha=1.0f/final_len)
```

- 用 ping-pong `basic_layer_output_` 在 stage 间传递（dim 翻倍、token 减半，所以
  buffer 大小不变）。
- **avgpool 用 GEMM 实现**：和一个全 1 向量（`avg_pool_ones_`，`deviceFill(1.0f)`）
  做 strided batched GEMM，`alpha = 1/final_len`，等价于对 token 维求平均。
- head（最终分类 Linear）在 `SwinTransformer::forward` 之外。

## 当前 tiny-cutlass 实现

**当前没有 BasicLayer 或整网这一层。** `csrc/swin/` 提供的是：

- `device::PatchEmbed<Kernel>`：独立的 patch embed（见 [PATCH_EMBED.md](PATCH_EMBED.md)）。
- `device::Swin<Kernel>`：一个 block 的 attention 子路径（见 [SWIN_BLOCK.md](SWIN_BLOCK.md)）。

没有：

| 整网/stage 组件 | 官方 | FT | 当前 cutlass | 状态 |
| --- | --- | --- | --- | --- |
| depth 循环 + shift 交替 | `BasicLayer.forward` | `BasicLayer::forward` | 无 | **缺失** |
| stage 间 ping-pong buffer | 隐式 | `basic_layer_output_` | 无 | **缺失** |
| PatchMerging 集成 | `downsample` | `patchMerge` | 仅 gather 件 | **缺失** |
| stage 串联（4 stage） | `self.layers` | basic layer loop | 无 | **缺失** |
| 最终 LayerNorm | `self.norm` | `invokeGeneralLayerNorm` | 无（但有可复用的 LN 件） | **缺失** |
| avgpool | `AdaptiveAvgPool1d` | strided GEMM + ones | 无 | **缺失** |
| head | `self.head` | 外部 | 无 | **缺失** |

`csrc/tests/swin/window.cpp` 是按**单 stage 配置逐个手动跑** attention 子路径来对
parity（`STATUS.md` 列了 stage0..stage3 的 benchmark 数字），并不是真的整网 driver
在串联。整网 logits parity 也还没作为 gate 通过（见 `STATUS.md`）。

## 移植建议（补齐整网）

依赖关系：先补齐单 block（[SWIN_BLOCK.md](SWIN_BLOCK.md)）和 PatchMerging
（[PATCH_MERGING.md](PATCH_MERGING.md)），再做 stage / 整网 driver。

```text
1. SwinBlock 完整版:  norm1 -> attn -> 残差1 -> norm2 -> MLP -> 残差2
2. BasicLayer driver: 一个 host 端循环，depth 次调用 block，shift 按 i 交替，
                      ping-pong buffer；结束调 PatchMerging（非末 stage）。
3. SwinModel driver:  patch_embed -> 4x BasicLayer（dim/res 按 stage 推进）->
                      最终 LayerNorm -> avgpool（可先用简单 reduce kernel，
                      或仿 FT 用 GEMM + ones）-> head Gemm。
```

driver 层适合做成模板 policy 风格的 host facade（参考现有 `device::Swin` /
`device::PatchEmbed`），把每个 stage 的 problem（dim、res、heads、shift）配好后
顺序 launch。权重和 workspace 用结构体集中管理，类似 FT 的 `SwinWeight` 层次。

验证策略：自底向上。先单 block 对官方 PyTorch parity，再单 stage（含 merge），
最后整网 logits 对官方 / cuDNN reference。

## FP8 注意点

- 整网每个 stage 的激活动态范围不同，FP8 量化 scale 需要 per-stage（甚至
  per-tensor / per-channel）标定。
- 最终 LayerNorm 和 avgpool 在高精度里做，head Gemm 可保持 FP16/FP32 输出。
- 先把整条 FP16 baseline 串通并验证，再逐 stage 替换成 FP8。
