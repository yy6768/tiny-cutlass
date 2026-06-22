# Swin 全流程概览

这个文档把 Swin Transformer 的完整前向流程拆成阶段，逐段对照三方实现：

- **官方 PyTorch**：`microsoft/Swin-Transformer` `models/swin_transformer.py`（Swin v1）。
- **FasterTransformer**：`NVIDIA/FasterTransformer` `src/fastertransformer/models/swin`（FP16 路径）。
- **当前 tiny-cutlass**：`csrc/swin/`。

每个阶段的细节文档：

| 阶段 | 文档 | 官方对应 | FT 对应 |
| --- | --- | --- | --- |
| PatchEmbed | [PATCH_EMBED.md](PATCH_EMBED.md) | `PatchEmbed` | `patchEmbed` (conv2d + AddBiasLayernorm) |
| Window 划分 | [../WINDOW_PARTITION.md](../WINDOW_PARTITION.md) | `window_partition` / `roll` | `invokeLayernormShiftPartition` |
| WindowAttention | [WINDOW_ATTENTION.md](WINDOW_ATTENTION.md) | `WindowAttention.forward` | `WindowAttention<T>::forward` |
| Swin Block | [SWIN_BLOCK.md](SWIN_BLOCK.md) | `SwinTransformerBlock.forward` | `SwinTransformerBlock<T>::forward` |
| PatchMerging | [PATCH_MERGING.md](PATCH_MERGING.md) | `PatchMerging.forward` | `invokeMergeLayernorm` + Gemm |
| BasicLayer / 整网 | [BASIC_LAYER_AND_MODEL.md](BASIC_LAYER_AND_MODEL.md) | `BasicLayer` / `SwinTransformer` | `SwinTransformerBasicLayer` / `SwinTransformer` |

## 类层次

官方和 FT 的类层次是一致的四层结构，权重结构镜像同一层次：

```text
SwinTransformer            整网：patch_embed -> stages -> norm -> avgpool -> head
  BasicLayer (每 stage 一个)  depth 个 block，结束做 PatchMerging（最后一个 stage 除外）
    SwinTransformerBlock      norm -> (shift) partition -> attn -> reverse -> residual -> MLP
      WindowAttention         QKV -> QK^T -> +relpos+mask -> softmax -> PV -> proj
```

Swin-T 默认配置（`img=224, patch=4, embed_dim=96, depths=[2,2,6,2],
heads=[3,6,12,24], window=7`）的逐 stage shape：

```text
image            [B, 3, 224, 224]
patch_embed   -> [B, 56*56=3136, 96]
stage0 (dim 96,  res 56) -> merge -> [B, 28*28, 192]
stage1 (dim 192, res 28) -> merge -> [B, 14*14, 384]
stage2 (dim 384, res 14) -> merge -> [B,  7*7,  768]
stage3 (dim 768, res 7)  (无 merge)  [B,  7*7,  768]
norm + avgpool -> [B, 768]
head          -> [B, num_classes]
```

每个 stage 的 token 数缩小到 1/4，channel 翻倍。

## 完整前向（官方 PyTorch）

`SwinTransformer.forward_features`：

```python
x = self.patch_embed(x)          # [B, L, C]
if self.ape: x = x + self.absolute_pos_embed
x = self.pos_drop(x)
for layer in self.layers:        # BasicLayer x4
    x = layer(x)
x = self.norm(x)                 # 最终 LayerNorm，[B, L, C]
x = self.avgpool(x.transpose(1, 2))  # [B, C, 1]
x = torch.flatten(x, 1)          # [B, C]
return self.head(x)              # [B, num_classes]
```

每个 `BasicLayer` 内：

```python
for blk in self.blocks:          # depth 个 block，shift 交替 0 / window//2
    x = blk(x)
if self.downsample is not None:  # PatchMerging
    x = self.downsample(x)
```

每个 `SwinTransformerBlock` 内（pre-norm v1）：

```python
shortcut = x
x = norm1(x)                     # 块前 LayerNorm
x = roll(x, -shift)              # SW-MSA 才有
x = window_partition(x)          # [nW*B, ws*ws, C]
x = attn(x, mask=attn_mask)      # WindowAttention
x = window_reverse(x)
x = roll(x, +shift)              # 反向 shift
x = shortcut + drop_path(x)      # 残差 1
x = x + drop_path(mlp(norm2(x))) # 残差 2 + MLP
```

## 完整前向（FasterTransformer FP16）

`SwinTransformer<T>::forward`：

```text
patchEmbed(conv2d + invokeAddBiasLayernorm)
loop basic layers (ping-pong buffer, do_patch_merge=层数>1 且非最后一层)
invokeGeneralLayerNorm  (最终 norm)
avgpool 用 stridedBatchedGemm 与全 1 向量相乘实现，alpha=1/len
```

`SwinTransformerBasicLayer<T>::forward`：

```text
loop depth blocks:
    shift_size = (i % 2 == 0) ? 0 : window/2
    block.forward
patchMerge:  invokeMergeLayernorm(concat 4 邻 + LN) -> Gemm 4C->2C
```

`SwinTransformerBlock<T>::forward`（v1）：

```text
invokeLayernormShiftPartition           # LN + shift + partition 融合
atten_->forward                          # WindowAttention
invokeGeneralAddBiasResidualPreLayerNorm # attn 输出 +bias +残差，并产出 MLP 前 LN
Gemm fc1 -> invokeAddBiasGeluV2 -> Gemm fc2
invokeAddBiasResidual                    # fc2 +bias +残差
```

FT 与官方的核心差异是**融合粒度**：官方是一串独立的 PyTorch 算子，FT 把
`LN+shift+partition`、`bias+residual+LN`、`bias+GELU`、`reverse+roll` 都各融成
一个 kernel，并把 attention 的 `scale+relpos+mask+softmax` 融成
`invokeMaskedSoftMaxWithRelPosBias`。

## 当前 tiny-cutlass 的状态

当前 `csrc/swin/` 只实现了**一个 Swin Block 的 attention 子路径**加上
**独立的 PatchEmbed**，并不是整网。`device::Swin<Kernel>::run`（`swin.cu`）的实际
stage 序列是：

```text
threadblock::WindowPartition      # shift 融进 index mapping，但没有 LN
cutlass::gemm::device::Gemm       # QKV projection
threadblock::AddQkvBiasSplit      # 加 QKV bias，拆成 Q/K/V
AttentionKernel (flash-attn 02)   # QK -> bias/mask -> online softmax -> PV
cutlass::gemm::device::Gemm       # output projection
threadblock::AddBias              # 加 output bias
threadblock::WindowReverse        # 反 partition + 反 shift
threadblock::PatchMerge           # 可选，仅 gather，不含 LN/Linear
```

`device::PatchEmbed<Kernel>`（`patch_embed.cu`）是另一条独立路径，见
[PATCH_EMBED.md](PATCH_EMBED.md)。

### 已经覆盖的

- PatchEmbed：conv2d + bias + LayerNorm，已对 host reference parity（`MAE≈1.4e-4`）。
- WindowPartition / WindowReverse：shift 融进坐标映射，已对 parity。
- QKV / output projection：用 `cutlass::gemm::device::Gemm`。
- WindowAttention 核心：复用 flash-attention 02 的 tiled online attention，
  relpos bias 和 mask 作为 `attn_bias` 预融合传入；stage2 shifted parity `MAE=0`。
- 相对位置偏置 + mask：以 `attention_bias [BW, heads, L, L_pad]` 形式预计算好喂进去。

### 与整网相比缺失的（关键 gap）

下面这些是 FT/官方有、但当前 cutlass 路径还**没有**的部分，也是后续移植的工作面：

1. **块前 LayerNorm（norm1）**：当前 `WindowPartition` 只做 layout copy，没有把
   官方 `norm1` / FT `invokeLayernormShiftPartition` 的 LN 融进来。见
   [SWIN_BLOCK.md](SWIN_BLOCK.md)。
2. **两处残差**：attention 残差和 MLP 残差都没有。当前 `run` 在 `WindowReverse`
   后就结束，没有 `shortcut + x`。
3. **MLP / FFN**：fc1 -> GELU -> fc2 完全缺失。
4. **norm2**：MLP 前 LayerNorm 缺失。
5. **PatchMerging 的 LN + Linear**：当前 `PatchMerge` 只做 2x2 gather 拼成 `4C`，
   没有 `norm(4C)` 和 `reduction(4C->2C)`。见 [PATCH_MERGING.md](PATCH_MERGING.md)。
6. **stage 串联 / 整网 driver**：没有 BasicLayer 的 depth 循环、shift 交替、
   ping-pong buffer，也没有最终 norm + avgpool + head。见
   [BASIC_LAYER_AND_MODEL.md](BASIC_LAYER_AND_MODEL.md)。
7. **QKV bias 之外的 scale**：当前 scale 通过 `problem.scale` 传给 attention kernel，
   需确认与官方 `(dim//heads)**-0.5` 一致（见 [WINDOW_ATTENTION.md](WINDOW_ATTENTION.md)）。

### 设计风格约束

当前 stage 是「拆开、易验证」的形态，不是 FT 的最终融合形态。公共 API 不暴露把
dtype / arch / layout 写进名字的 concrete alias；只在显式实例化、测试、launcher 层
出现具体 `Sm80 + half_t` 选择。详见各阶段文档的「当前实现」与「问题」小节。

## FP8 升级方向

目标是把 FT 那一代的 INT8 量化升级成 FP8（E4M3, SM89）。当前 cutlass 路径是纯 FP16
baseline，FP8 作为后续编号变体。关键注意点（详见各阶段文档）：

- FP8 E4M3 max≈448，mask 的 `-100` 偏置必须在高精度里加，不能存成 E4M3。
- COL32 是 INT8 MMA 的 layout artifact，FP8 TensorOp on SM89 用不同 layout，
  应直接用 CUTLASS 3.x FP8 GEMM + scale epilogue，而不是搬 INT8 的 COL32 机制。
- 先把 FP16 baseline 的每个阶段补全并验证，再做 FP8。
