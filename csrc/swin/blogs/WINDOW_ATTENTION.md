# Swin WindowAttention

这个文档记录 Swin WindowAttention 的三方实现：官方 PyTorch、FasterTransformer
FP16，以及当前 tiny-cutlass。WindowAttention 是把已经划分好的 window token 做
multi-head self-attention，并加上相对位置偏置和（SW-MSA 时的）attention mask。

## 对齐的语义

输入是 partition 后的 window token：

```text
input  : [nW*B, N, C]      N = window_size * window_size, C = dim
output : [nW*B, N, C]
```

其中：

```text
nW         = (H / window_size) * (W / window_size)
num_heads  = head_number
head_dim   = C / num_heads
scale      = head_dim ** -0.5
```

数学路径：

```text
qkv   = Linear_C->3C(x) + bias            拆成 q, k, v: [nW*B, heads, N, head_dim]
q     = q * scale
attn  = q @ k^T                            [nW*B, heads, N, N]
attn += relative_position_bias             [heads, N, N] broadcast
attn += attn_mask                          仅 SW-MSA，[nW, N, N] broadcast
attn  = softmax(attn, dim=-1)
out   = attn @ v                           [nW*B, heads, N, head_dim]
out   = out 转置回 [nW*B, N, C]
out   = Linear_C->C(out) + bias            output projection
```

## 官方 PyTorch 实现

```python
def forward(self, x, mask=None):
    B_, N, C = x.shape
    qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    q = q * self.scale
    attn = (q @ k.transpose(-2, -1))

    relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, N, N
    attn = attn + relative_position_bias.unsqueeze(0)

    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
    else:
        attn = self.softmax(attn)

    attn = self.attn_drop(attn)
    x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x
```

要点：

- `scale = (dim // num_heads) ** -0.5`，**乘在 q 上**，在 matmul 之前。
- `qkv` reshape 成 `[B_, N, 3, nH, head_dim]` 再 permute 成 `[3, B_, nH, N, head_dim]`，
  所以 QKV 权重布局是 `[C, 3*C]`，每个 token 的 3C 输出按 `(qkv, head, head_dim)`
  顺序排列。
- softmax 在最后一维（key 维）。
- output projection 是 `Linear(dim, dim)`，带 bias。

### 相对位置偏置

```python
self.relative_position_bias_table = nn.Parameter(
    torch.zeros((2*Wh-1) * (2*Ww-1), num_heads))    # 表 shape

coords = torch.stack(torch.meshgrid([arange(Wh), arange(Ww)]))  # 2, Wh, Ww
coords_flatten = torch.flatten(coords, 1)                       # 2, N
relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, N, N
relative_coords = relative_coords.permute(1, 2, 0).contiguous()           # N, N, 2
relative_coords[:, :, 0] += Wh - 1
relative_coords[:, :, 1] += Ww - 1
relative_coords[:, :, 0] *= 2*Ww - 1
relative_position_index = relative_coords.sum(-1)              # N, N, 值域 [0, (2Wh-1)(2Ww-1))
```

- `relative_position_bias_table` shape `((2Wh-1)(2Ww-1), nH)`。
- `relative_position_index` shape `(N, N)`，把二维相对坐标压成一维 index：行轴
  相对坐标先 `*(2Ww-1)` 再和列轴相加。
- 查表得到 `(N, N, nH)`，permute 成 `(nH, N, N)`，broadcast 加到每个 batch。

### SW-MSA attention mask

mask 在 `SwinTransformerBlock` 里构造（见 [SWIN_BLOCK.md](SWIN_BLOCK.md)）：

```python
attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, N, N]
attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
```

同 region 的 token 间是 `0`，跨 region 是 `-100.0`（注意不是 `-inf`）。

## FasterTransformer 实现

`WindowAttention<T>::forward` 有两条路径，靠 `use_trt_` 选择（`sm∈{75,80,86}` 且
`size_per_head==32` 且 `window_len<=TRT_MAX_LEN` 且 `T==half`）。

### TRT-fused 路径

```text
Gemm QKV (fused bias)               # [3*dim], CUDART>=11
dispatcher_fp16_->setup(S, batch*window_num, window_num)
dispatcher_fp16_->run(q_buf_, mask, trt_relative_position_bias, window_len, qkv_buf_)
Gemm output proj                    # attention_output_weight.kernel
```

mask 只在 `shift_size != 0` 时传 `trt_attention_mask`，否则 `nullptr`。

### 非融合路径

```text
Gemm QKV                            # qkv_buf_ = [m, head*3*size]
invokeAddHead3SizeQKVBias           # 加 bias，拆成 q/k/v_buf_ [B*nW, heads, N, size]
stridedBatchedGemm QK^T             # qk_buf_ [B, nW, heads, N, N], batch=B*nW*heads
invokeMaskedSoftMaxWithRelPosBias   # scale + relpos + mask + softmax 全融合
stridedBatchedGemm attn*V           # qkv_buf_
invokeTransposeQKV                  # [B, head, len, size] -> [B, len, head, size]
Gemm output proj
```

两条路径最后都调 `invokeReverseRoll`（见 [SWIN_BLOCK.md](SWIN_BLOCK.md)）。

### invokeMaskedSoftMaxWithRelPosBias 的索引

这是 FT 把 scale/relpos/mask/softmax 融成一个 kernel 的核心。在
`qk_buf [batch, window_num, num_head, window_len, window_len]` 上原地操作，
grid `(window_len, window_num*num_head, batch)`：

```c
offset_in_window         = window_id*window_len + threadIdx.x;        // (q_row, k_col)
qk_offset                = (blockIdx.z*gridDim.y + blockIdx.y)*window_len^2 + offset_in_window;
relative_pos_bias_offset = (blockIdx.y % num_head)*window_len^2 + offset_in_window;  // 每 head 一份 bias
mask_offset              = (blockIdx.y / num_head)*window_len^2 + offset_in_window;  // 每 window 一份 mask
tmp = qk_scale * qk_buf[qk_offset] + mask_val + relative_pos_bias[relative_pos_bias_offset];
// block softmax: max-reduce -> exp(tmp - s_max) -> sum-reduce(+1e-6) -> normalize
```

`blockIdx.y % num_head` 选 head（relpos bias 按 head 索引），`blockIdx.y / num_head`
选 window（SW-MSA mask 按 window 索引）；`mask` 为 `nullptr` 时取 0。

### invokeAddHead3SizeQKVBias 的索引

加 QKV bias 并把交错的 `[m, head*3*size]` 拆成三个转置好的 buffer，grid
`(window_len, window_num, 3*batch)`：

```c
qkv_id    = blockIdx.z / batch;     // 0=Q,1=K,2=V
head_id   = threadIdx.x / size_per_head;
bias_idx  = (head_id*3 + qkv_id)*size_per_head + (threadIdx.x % size_per_head);
input_idx = ((batch_id*window_num + window_id)*window_len + token_id)*num_head*3*size + bias_idx;
target_id = (((batch_id*window_num+window_id)*num_head + head_id)*window_len + token_id)*size + id_in_head;
buf_ptr[target_id] = mm_qkv[input_idx] + bias;
```

## 当前 tiny-cutlass 实现

入口在 `swin.cu` 的 `launch_projection`（QKV/output proj）、`AddQkvBiasSplit`
（threadblock）和 `launch_attention`（复用 flash-attention 02）。序列：

```text
cutlass::gemm::device::Gemm           # QKV projection [rows, C] x [C, 3C]
threadblock::AddQkvBiasSplit          # 加 QKV bias，拆 q/k/v
AttentionKernel (flash-attn 02)       # QK -> +attn_bias -> online softmax -> PV
cutlass::gemm::device::Gemm           # output projection [rows, C] x [C, C]
threadblock::AddBias                  # 加 output bias
```

### QKV projection

`launch_projection<Kernel>` 用 `cutlass::gemm::device::Gemm`，
`{rows, 3C, C}`，alpha=1/beta=0，没有 epilogue bias：

```cpp
typename Gemm::Arguments args(
    {rows, n, k},          // rows = swin_rows(problem), k = C, n = 3C
    {input, k},            // windows [rows, C]
    {weight, n},           // qkv_weight [C, 3C]
    {output, n},           // qkv [rows, 3C]
    {output, n},
    {ElementCompute(1), ElementCompute(0)});
```

### AddQkvBiasSplit（threadblock）

每个线程处理一个 `(row, channel)`，从 `qkv [rows, 3C]` 里取 q/k/v 三段，加上
对应 bias 段，写到三个独立 buffer：

```cpp
int c   = idx % channels;
int row = idx / channels;
int64_t base = int64_t(row) * (3 * channels);
Element qv = qkv[base + c];
Element kv = qkv[base + channels + c];
Element vv = qkv[base + 2*channels + c];
// + bias[c] / bias[channels+c] / bias[2*channels+c]
q[idx] = qv; k[idx] = kv; v[idx] = vv;
```

注意这里的 QKV 输出布局假设是 `[..., (qkv, channel)]`，即整段 q 在前、k 在中、
v 在后，channel 连续。这和官方 `qkv.reshape(B_,N,3,nH,head_dim)` 的
`(3, head, head_dim)` 顺序一致（channel = head*head_dim）。

### WindowAttention 核心（flash-attention 02）

`launch_attention<Kernel>` 直接复用工程里的 `AttentionKernel`（flash-attention
02 tiled online attention），一个 kernel 内完成 `QK -> bias/mask -> online
softmax -> PV`，不落全局 attention matrix：

```cpp
params.scale          = problem.scale;       // 外部传入的 head_dim**-0.5
params.attn_bias_ptr  = tensors.attention_bias;  // relpos + mask 预融合
params.num_queries    = l;   params.num_keys = l;   // l = window_len
params.head_dim       = problem.head_size;
params.num_heads      = problem.head_number;
params.num_batches    = bw;  // batched windows = B * nW
params.bias_strideM   = lp;  // L_pad，对齐到 8
params.bias_strideH   = int64_t(l) * lp;
params.bias_strideB   = int64_t(head_number) * l * lp;
```

关键设计差异：

- relpos bias 和 SW-MSA mask 不在 kernel 里分别加，而是**预先合并成一个
  `attention_bias [BW, heads, L, L_pad]`**，作为 flash-attention 的 `attn_bias`
  传入。这把 FT 的 `invokeMaskedSoftMaxWithRelPosBias` 的 relpos+mask 部分搬到了
  host/预处理，softmax 仍在 kernel 内 online 完成。
- `L_pad = swin_window_len_padded`，把 `L` 对齐到 8（`((l+7)/8)*8`），满足 bias
  tensor 的对齐要求。
- scale 通过 `problem.scale` 注入，由调用方负责设成 `head_dim ** -0.5`。
- QKV 已经被 `AddQkvBiasSplit` 拆成独立的 `query/key/value` buffer，布局
  `[BW, L, heads, head_dim]`，stride 见 `swin.cu`。

### output projection + bias

第二个 `launch_projection` 做 `[rows, C] x [C, C]`，随后 `threadblock::AddBias`
按 channel 加 output bias。注意当前没有在这一步融合残差。

## 当前路径和官方 / FT 的差异

| 步骤 | 官方 | FT | 当前 cutlass |
| --- | --- | --- | --- |
| QKV | `qkv` Linear+bias | Gemm (+fused bias / AddHead3SizeQKVBias) | Gemm + AddQkvBiasSplit |
| scale | `q * scale` | softmax kernel 里乘 | flash-attn `params.scale` |
| QK^T | `q @ k^T` | stridedBatchedGemm | flash-attn 内部 |
| relpos+mask | 两次 broadcast add | `invokeMaskedSoftMaxWithRelPosBias` | 预融合成 `attention_bias` |
| softmax | `nn.Softmax(-1)` | 同上 kernel | flash-attn online softmax |
| attn·V | `attn @ v` | stridedBatchedGemm | flash-attn 内部 |
| transpose | `.transpose(1,2)` | `invokeTransposeQKV` | flash-attn 输出已是 token-major |
| proj | `proj` Linear+bias | Gemm + bias | Gemm + AddBias |

当前路径相当于把 FT 的非融合路径里 `QK -> relpos/mask -> softmax -> PV ->
transpose` 这一整段，替换成 flash-attention 02 的 tiled online attention，并把
relpos/mask 预融合进 bias。优势是不落全局 attention matrix、显存友好；代价是
relpos+mask 的拼装放到了外部。

## 当前验证约束

- dtype：FP16 input/weight/output，FP32 accumulation/compute。
- `C = head_number * head_size`，`head_size % 8 == 0`，`C % 8 == 0`。
- `window_len <= 64`。
- `scale > 0`（必须由调用方设成 `head_dim ** -0.5`）。
- 需要 Ampere 或更新架构；当前显式实例化是 `Sm80 + half_t`。

stage2 shifted attention 对 cuDNN reference `MAE=0`，tail stage `MAE≈1e-9`。

## 已知问题 / 待办

1. **scale 一致性**：`problem.scale` 必须由调用方算成 `head_dim ** -0.5`；
   `swin_problem.h` 默认是 `0.0f`，`validate_problem` 会拒绝 `scale<=0`，但没有
   自动从 `head_size` 推导。整网串联时要确保每个 stage 都正确设置。
2. **relpos+mask 预融合的成本**：`attention_bias [BW, heads, L, L_pad]` 需要外部
   预计算并占显存；不同 shift 的 mask 不同，整网每个 SW-MSA block 都要一份。
   FT 是按 head 共享 relpos、按 window 共享 mask，当前是全展开。
3. **没有残差融合**：output proj 后只有 `AddBias`，缺 block 级残差（见
   [SWIN_BLOCK.md](SWIN_BLOCK.md)）。
4. **FP8 注意点**：mask 的 `-100` 偏置必须在高精度里加，不能存成 E4M3
   （max≈448 时 -100 可表示，但与 relpos 求和及 softmax 数值范围需在 FP32 累加）。
   QKV/proj GEMM 换成 CUTLASS FP8 GEMM + scale epilogue。
