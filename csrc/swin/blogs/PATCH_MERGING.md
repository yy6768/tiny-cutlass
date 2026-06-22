# Swin PatchMerging

这个文档记录 `PatchMerging` 的三方实现。PatchMerging 是 stage 之间的 downsample：
把 2x2 空间邻域拼到 channel 维（H,W 各减半，channel 变 4C），LayerNorm，再用一个
无 bias 的 Linear 把 `4C` 压到 `2C`。

## 对齐的语义

```text
input  : [B, H*W, C]
output : [B, (H/2)*(W/2), 2C]
```

数学路径：

```text
x  -> view [B, H, W, C]
x0 = x[:, 0::2, 0::2, :]      # 偶 H, 偶 W
x1 = x[:, 1::2, 0::2, :]      # 奇 H, 偶 W
x2 = x[:, 0::2, 1::2, :]      # 偶 H, 奇 W
x3 = x[:, 1::2, 1::2, :]      # 奇 H, 奇 W
cat[x0,x1,x2,x3] -> [B, H/2, W/2, 4C]
view -> [B, (H/2)*(W/2), 4C]
norm(4C) -> reduction Linear 4C->2C (bias=False)
```

**拼接顺序极其关键**（决定权重对齐）：channel 维按 `x0,x1,x2,x3` 排，每段是
完整的 `C`。第一个 slice 轴是 H（行），第二个是 W（列）。

## 官方 PyTorch 实现

```python
class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]   # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]   # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]   # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]   # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)
        return x
```

要点：

- `norm` 是 LayerNorm over `4C`，在 reduction **之前**。
- `reduction` 是 `Linear(4C, 2C, bias=False)`，**没有 bias**。
- 4 个邻居的顺序是 `(even,even), (odd,even), (even,odd), (odd,odd)`，索引
  `[:, h_slice, w_slice, :]` 里第一个是 H、第二个是 W。

## FasterTransformer 实现

在 `SwinTransformerBasicLayer<T>::patchMerge`（v1）：

```text
invokeMergeLayernorm(merge_buf, input, gamma, beta, eps, batch, res, res, dim)
Gemm(T, N, m=2*dim, n=batch*res*res/4, k=4*dim,
     A=weight (4*dim), B=merge_buf (4*dim), C=output (2*dim))
```

`invokeMergeLayernorm` 把 2x2 邻域 concat（channel -> `4*dim`）和 LayerNorm
融成一个 kernel，然后一个 Gemm 做 `4C -> 2C`。

### invokeMergeLayernorm 的索引

grid `(W/2, H/2, batch)`，对输出 channel `col_id ∈ [0, 4n)`（`n = orig_dim`）：

```c
n_4         = n >> 2;            // 输入到 kernel 的 n 已是 4*orig，n_4 = orig_dim
part_id     = col_id / n_4;      // 4 个邻居中的哪一个 (0..3)
offset_in_W = part_id / 2;       // 0 或 1
offset_in_H = part_id % 2;       // 0 或 1
input_id    = batch_offset
            + (2*H_idx + offset_in_H) * input_H_stride
            + (2*W_idx + offset_in_W) * n_4
            + (col_id % n_4);
// LayerNorm over 4*orig channels
out[batch_offset + H_idx*output_H_stride + W_idx*(4*orig) + col_id]
```

`part_id` 从 0 到 3 对应 `offset (H,W) = (0,0),(1,0),(0,1),(1,1)`，即
`(even,even),(odd,even),(even,odd),(odd,odd)`，与官方 `cat([x0,x1,x2,x3])` 的
顺序一致。

v2 顺序不同：`invokeImageMerge -> Gemm -> LayerNorm`（post-norm）。

## 当前 tiny-cutlass 实现

入口在 `threadblock::PatchMerge`（`threadblock/layout.h`）和 `warp::
patch_merge_input_token`（`warp/layout.h`）。`swin.cu` 里作为可选 stage 调用。

```cpp
// threadblock::PatchMerge::run
int64_t output_base = (b*out_h*out_w + oh*out_w + ow) * (4 * channels);
for (int col = threadIdx.x; col < 4 * channels; col += blockDim.x) {
    int64_t input_idx = warp::patch_merge_input_token(pixel, height, width, channels, col);
    output[output_base + col] = input[input_idx];   // 纯 gather
}
```

```cpp
// warp::patch_merge_input_token
int part = column / channels;     // 0..3
int c    = column - part*channels;
int offset_w = part / 2;          // 0 或 1
int offset_h = part % 2;          // 0 或 1
int y = 2*oh + offset_h;
int x = 2*ow + offset_w;
return ((b*height*width + y*width + x) * channels) + c;
```

### 当前实现只做 gather

`offset (h,w)` 随 `part = 0..3` 取 `(0,0),(1,0),(0,1),(1,1)`，和官方/FT 的
`x0..x3` 拼接顺序一致。但当前 `PatchMerge` **只做 2x2 -> 4C 的 gather**：

```text
[B, H, W, C] -> [B, H/2, W/2, 4C]    // 仅拼接，无 LN，无 Linear
```

对照完整 PatchMerging 缺失：

| 步骤 | 官方 | FT | 当前 cutlass | 状态 |
| --- | --- | --- | --- | --- |
| 2x2 concat -> 4C | `cat([x0..x3])` | 融进 `MergeLayernorm` | `PatchMerge` gather | 有 |
| LayerNorm(4C) | `self.norm` | 融进 `MergeLayernorm` | 无 | **缺失** |
| Linear 4C->2C | `self.reduction`(no bias) | Gemm 4C->2C | 无 | **缺失** |

注意当前 `swin.cu` 里 `PatchMerge` 的输入是 `tensors.input`（block 的原始输入），
作为 stage transition 的实验性 gather，并没有接 LayerNorm + reduction，也没有真正
串进 stage 输出。`STATUS.md` 提到「tail stage 不应误触发 patch merge」，说明它目前
是一个 optional、未完全集成的件。

## 移植建议（补齐 PatchMerging）

```text
1. concat: 复用现有 PatchMerge gather，产出 [B, H/2, W/2, 4C]。
2. norm:   复用 AddBiasLayerNorm（去掉 bias，channel = 4C），或新增独立 LN。
           可进一步把 gather+LN 融成一个 kernel 对齐 FT 的 MergeLayernorm。
3. reduce: cutlass::gemm::device::Gemm，[N, 4C] x [4C, 2C]，无 bias
           (beta=0，epilogue 不加 bias)。复用 launch_projection 模式。
```

验证：对官方单个 PatchMerging 比较 `[B,(H/2)(W/2),2C]` 的 MAE。要特别核对
拼接顺序和 reduction 权重的 `4C` 维排布是否和官方 `cat` 顺序一致。

## 支持约束

- `H % 2 == 0` 且 `W % 2 == 0`（当前 `swin.cu` 已检查 `image_size % 2`）。
- 拼接顺序固定 `(even,even),(odd,even),(even,odd),(odd,odd)`。
- reduction 无 bias。

## FP8 注意点

- LayerNorm 在 FP32 reduce 后再量化到 FP8。
- reduction GEMM 用 CUTLASS FP8 GEMM + scale epilogue；输入是 4C concat 的结果，
  量化 scale 需覆盖 4 个邻域拼起来的动态范围。
