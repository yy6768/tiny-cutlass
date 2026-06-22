# Swin Block

这个文档记录 `SwinTransformerBlock` 的三方实现。Block 是 Swin 的基本计算单元，
把一次 (S)W-MSA attention 和一个 MLP 用两处残差和两处 LayerNorm 串起来。

## 对齐的语义

输入输出都是 token 序列：

```text
input  : [B, L, C]      L = H * W
output : [B, L, C]
```

Swin v1 是 **pre-norm**：LayerNorm 在 attention/MLP 之前，残差跨过 norm。一个
block 的数学路径：

```text
shortcut = x
x = norm1(x)
x = shift + window_partition(x)          # SW-MSA 才 shift
x = WindowAttention(x, mask)
x = window_reverse(x) + reverse_shift
x = shortcut + drop_path(x)              # 残差 1
x = x + drop_path(MLP(norm2(x)))         # 残差 2
```

其中 `MLP(y) = fc2(GELU(fc1(y)))`，`fc1: C->4C`，`fc2: 4C->C`。

W-MSA（`shift_size==0`）和 SW-MSA（`shift_size==window//2`）的区别只在
shift 和 attention mask。`BasicLayer` 里 block 按 `i` 偶/奇交替这两种。

## 官方 PyTorch 实现

```python
def forward(self, x):
    H, W = self.input_resolution
    B, L, C = x.shape
    shortcut = x
    x = self.norm1(x)
    x = x.view(B, H, W, C)

    # cyclic shift
    if self.shift_size > 0:
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        x_windows = window_partition(shifted_x, self.window_size)
    else:
        shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
    x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

    # W-MSA / SW-MSA
    attn_windows = self.attn(x_windows, mask=self.attn_mask)

    # merge windows + reverse shift
    attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
    if self.shift_size > 0:
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
    else:
        x = window_reverse(attn_windows, self.window_size, H, W)
    x = x.view(B, H * W, C)
    x = shortcut + self.drop_path(x)             # 残差 1

    # FFN
    x = x + self.drop_path(self.mlp(self.norm2(x)))  # 残差 2
    return x
```

要点：

- `shortcut` 是 **norm1 之前** 的原始输入，残差 1 跨过 norm1+attention。
- 残差 2 跨过 norm2+MLP。
- `roll` 用 `-shift_size` 做 cyclic shift，attention 后用 `+shift_size` 还原。
- `drop_path`（stochastic depth）推理时是 identity。

### attention mask 的构造（SW-MSA）

mask 在 block 构造时一次算好（依赖 `input_resolution` 和 `shift_size`）：

```python
if self.shift_size > 0:
    H, W = self.input_resolution
    img_mask = torch.zeros((1, H, W, 1))
    h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size).view(-1, window_size*window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
else:
    attn_mask = None
```

- `img_mask` 用 3x3 的 slice 组合给 9 个区域打 0..8 的编号。
- partition 后同 window 内，相同编号的 token 之间 mask=0（允许 attend），不同
  编号 mask=-100（cyclic shift 把不相邻的图像区域拼进同一 window，需要屏蔽）。

## FasterTransformer 实现

`SwinTransformerBlock<T>::forward`（v1）：

```text
window_size_in_use = min(input_resolution, window_size)   # 小分辨率退化
invokeLayernormShiftPartition(normed_shifted_input_, input, gamma1, beta1, eps,
                              batch, res, res, dim, shift_size, window_size_in_use)
atten_->forward(...)                                       # -> attention_output_
invokeGeneralAddBiasResidualPreLayerNorm(attention_output_, normed_attn_out_buf_,
                              attention_output_, input, gamma2, beta2,
                              attn_output_weight.bias, eps, batch*res*res, dim)
Gemm fc1     (intermediate_weight.kernel)  -> mlp_buf_
invokeAddBiasGeluV2(mlp_buf_, intermediate_weight.bias, batch*res*res, mlp_dim)
Gemm fc2     (output_weight.kernel)        -> output
invokeAddBiasResidual(output, attention_output_, output_weight.bias, batch*res*res, dim)
```

FT 的融合策略：

1. **`invokeLayernormShiftPartition`**：把 `norm1 + cyclic shift + window
   partition` 融成一个 kernel（见 [../WINDOW_PARTITION.md](../WINDOW_PARTITION.md)
   的索引数学）。输出 `normed_shifted_input_` 直接给 attention。
2. **`invokeGeneralAddBiasResidualPreLayerNorm`**：一次 kernel 做三件事——给
   attention 输出加 bias、加残差 `input`（残差 1）、再算 MLP 前 LayerNorm
   （norm2）。残差结果留在 `attention_output_` 里，归一化结果写 `normed_attn_out_buf_`。
3. **`invokeAddBiasGeluV2`**：fc1 后融合 bias + GELU。
4. **`invokeAddBiasResidual`**：fc2 后融合 bias + 残差（残差 2，残差源是
   `attention_output_`）。

attention 内部的 reverse-roll 在 `WindowAttention::forward` 末尾用
`invokeReverseRoll` 完成（见 [WINDOW_ATTENTION.md](WINDOW_ATTENTION.md)），所以
block 这一层看到的是 attention 已经 reverse 回 `[batch, H, W, dim]` 的结果。

v2 不同：用 `invokeShiftPartition`（不带 LN）做 post-norm，残差用
`invokeAddBiasLayernormAddRes`。

## 当前 tiny-cutlass 实现

当前 `device::Swin<Kernel>::run`（`swin.cu`）**只覆盖了 block 里 attention 的
layout + projection 子路径**，不是完整 block：

```text
threadblock::WindowPartition      # 只做 shift+partition 的 layout copy，没有 norm1
cutlass::gemm::device::Gemm       # QKV projection
threadblock::AddQkvBiasSplit
AttentionKernel (flash-attn 02)
cutlass::gemm::device::Gemm       # output projection
threadblock::AddBias              # output bias，没有残差
threadblock::WindowReverse        # reverse + reverse shift
threadblock::PatchMerge           # 可选 gather
```

对照官方/FT 的完整 block，当前实现对应的只有：

```text
[缺 norm1] -> shift+partition -> QKV -> attn -> proj -> [缺残差] -> reverse
```

### 与完整 block 相比缺失的部分

| block 步骤 | 官方 | FT | 当前 cutlass | 状态 |
| --- | --- | --- | --- | --- |
| norm1（块前 LN） | `norm1(x)` | 融进 `LayernormShiftPartition` | 无 | **缺失** |
| shift + partition | `roll`+`window_partition` | `LayernormShiftPartition` | `WindowPartition`（含 shift） | 有（无 LN） |
| WindowAttention | `attn(...)` | `atten_->forward` | flash-attn 02 路径 | 有 |
| reverse + reverse shift | `window_reverse`+`roll` | `invokeReverseRoll` | `WindowReverse` | 有 |
| 残差 1（attn） | `shortcut + x` | `GeneralAddBiasResidualPreLayerNorm` | 无 | **缺失** |
| norm2（MLP 前 LN） | `norm2(x)` | 同上 kernel | 无 | **缺失** |
| MLP fc1 | `mlp.fc1` | Gemm | 无 | **缺失** |
| GELU | `mlp.act` | `AddBiasGeluV2` | 无 | **缺失** |
| MLP fc2 | `mlp.fc2` | Gemm | 无 | **缺失** |
| 残差 2（MLP） | `x + mlp(...)` | `AddBiasResidual` | 无 | **缺失** |

也就是说，当前 `swin.cu` 是一个**可验证的 attention 子路径骨架**，把 norm、
残差、整个 MLP 都留给了后续移植。`STATUS.md` 里跑通的 parity 是针对这个子路径
和 cuDNN attention reference 的，不是整 block。

### 现有 threadblock 复用基础

要补齐完整 block，下面这些已有的 `threadblock` 件可以直接复用或改造：

- `AddBiasLayerNorm`（`threadblock/layout.h`）：已经实现「加 bias + 每 token 对
  channel 维 LayerNorm」，PatchEmbed 在用。norm1/norm2 可以基于它扩展（norm1 还
  需要融合 shift+partition 才能对齐 FT；最简单的做法是先拆成独立 LN kernel）。
- `AddBias`：output bias，可扩展成 `AddBiasResidual`。
- QKV/output projection 的 `launch_projection` 模式可以直接复制给 MLP 的 fc1/fc2
  （只是 N 维不同：fc1 是 `C->4C`，fc2 是 `4C->C`）。
- GELU 目前没有 threadblock 件，需要新增一个 `AddBiasGelu`。

## 移植建议（补齐完整 block）

最小可验证路径，保持当前「拆开、易对 parity」的风格，不强求 FT 的融合：

```text
1. norm1:   新增独立 LayerNorm kernel（或复用 AddBiasLayerNorm 去掉 bias），
            产出 normed，再走现有 WindowPartition。或直接把 LN 融进 WindowPartition
            对齐 FT 的 LayernormShiftPartition。
2. attention: 现有路径（WindowPartition -> QKV -> attn -> proj -> AddBias -> WindowReverse）
3. 残差 1:  新增 AddResidual，把 reverse 后的结果加回 block 输入 shortcut。
4. norm2:   独立 LayerNorm。
5. MLP:     Gemm fc1 (C->4C) -> 新增 AddBiasGelu -> Gemm fc2 (4C->C) -> AddBias。
6. 残差 2:  AddResidual，加回残差 1 的结果。
```

验证时逐步加 stage，每加一段就和官方 PyTorch 单 block 对 MAE。

## FP8 注意点

- norm1/norm2 的 LayerNorm 输入是 attention/残差的累加值，范围较大，必须在 FP32
  里 reduce，再量化到 FP8 喂下一个 GEMM。
- mask 的 `-100` 见 [WINDOW_ATTENTION.md](WINDOW_ATTENTION.md)：在高精度里加。
- fc1/fc2 GEMM 换 CUTLASS FP8 GEMM + per-tensor/per-channel scale epilogue；
  GELU 在 FP32 里算再量化。FT 同代有 `layernorm_fp8_kernels.cu` 可借鉴。
