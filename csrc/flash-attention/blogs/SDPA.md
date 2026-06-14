# PyTorch SDPA 学习笔记：框架是怎么推动 attention 融合的

## 前言

我现在看 PyTorch 的 `scaled_dot_product_attention`，不是为了“直接用它就结束了”，而是为了弄清楚框架层对 attention fusion 的抽象边界。我们自己写 CUTLASS kernel 时，最终也要回答同一个问题：什么时候走自定义 kernel，什么时候让 PyTorch 选择 FlashAttention/cuDNN/mem-efficient/math backend，什么时候应该暴露一个可控的 dispatch path。

SDPA 的核心 API 是：

```python
torch.nn.functional.scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
)
```

它的语义仍然是：

$$
O = \text{softmax}(QK^T \cdot scale + bias) V
$$

但 PyTorch 的重点不在这行公式，而在它后面藏着一个 backend selector。

## SDPA 不是一个 kernel

PyTorch 官方文档目前列了三类 supported implementation：

1. FlashAttention-2；
2. Memory-Efficient Attention；
3. PyTorch C++ math implementation。

在 CUDA backend 上，SDPA 会根据输入尝试选择更快的 fused kernel；其他 backend 会走 PyTorch 实现。默认所有实现都 enable，PyTorch 会根据 dtype、shape、mask、dropout、GQA 等条件自动选择。

这说明一件事：SDPA 是一个语义入口，不是一个固定实现。如果我们拿 SDPA 当 reference，要记录当时实际选中的 backend，否则可能今天在一个 shape 上对齐的是 FlashAttention，另一个 shape 上对齐的是 math。

## 控制 backend 的方法

官方推荐用 `torch.nn.attention.sdpa_kernel` 这个 context manager 控制 backend：

```python
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out = scaled_dot_product_attention(q, k, v)

with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
    out = scaled_dot_product_attention(q, k, v)

with sdpa_kernel(
    [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION],
    set_priority=True,
):
    out = scaled_dot_product_attention(q, k, v)
```

`SDPBackend` 现在包括：

- `MATH`
- `FLASH_ATTENTION`
- `EFFICIENT_ATTENTION`
- `CUDNN_ATTENTION`
- `OVERRIDEABLE`
- `ERROR`

对我们来说，verify harness 里最好显式写出 reference backend。否则 benchmark 结果会很滑：你以为在和 FlashAttention 比，实际可能在和 math 或 cuDNN 比。

## 语义坑：dropout、mask、GQA

SDPA 文档里有几个很容易踩的地方。

第一，`dropout_p` 不会因为 module 进入 eval mode 自动关闭。你必须自己传 `0.0`。所以 correctness test 里默认应当固定 `dropout_p=0.0`，除非当前 variant 明确要实现 dropout。

第二，`attn_mask` 的 bool 语义和 `MultiheadAttention` 的 `key_padding_mask` 是反的。在 SDPA 里，`True` 表示参与 attention；在 MHA 的 padding mask 里，`True` 表示 masked out。写 reference 时如果 mask 方向错了，kernel 可能完全没错，但你会以为错。

第三，GQA 还是 experimental。官方文档写明它当前只在 CUDA 上的 FlashAttention 和 math kernel 里可用，并且要求：

- `num_heads_query % num_heads_key_value == 0`
- `num_heads_key == num_heads_value`

所以 GQA 不应该混在第一轮 tiny-cutlass attention verification 里。先把普通 MHA 做稳，再开 GQA family。

## PyTorch 的融合路线

我现在看到 PyTorch 的 attention acceleration 至少有三层。

第一层是 SDPA：给用户一个统一语义入口，让 backend selector 在 FlashAttention-2、Memory-Efficient、math、cuDNN 之间选。

第二层是 `sdpa_kernel`：允许用户或测试强制选择 backend。这个对我们很重要，因为我们要用它构造稳定 reference。

第三层是 FlexAttention：PyTorch 用 `score_mod`、`mask_mod`、`BlockMask` 和 `torch.compile` 把自定义 attention 变体降到 fused FlashAttention-like kernel，而不是让用户手写一堆 Triton/CUDA kernel。

FlexAttention 的关键思想是：

- `score_mod(score, b, h, q_idx, kv_idx)` 描述 score 上的逐元素修改；
- `mask_mod(b, h, q_idx, kv_idx)` 描述可稀疏跳过的 mask；
- `BlockMask` 把 mask 变成 block-level metadata，让 kernel 可以跳过完全 masked 的 block；
- `torch.compile` 把这些 Python-level 描述降成 fused kernel。

这个路线和我们写 CUTLASS kernel 的路线不冲突。它反而提醒我们：如果自定义 attention 变体只是 bias/mask/softcap 这种 score-level 改动，框架 codegen 可能更合适；如果是 tile policy、SMEM layout、WGMMA/TMA/FP8 pipeline 这种硬件路径，才更适合 tiny-cutlass 手写。

## tiny-cutlass 里的 reference 策略

我建议后续 tests 里明确分三种 reference：

1. `reference_math`：用 PyTorch 普通 matmul + softmax，最慢但语义最透明；
2. `reference_sdpa_forced`：用 `sdpa_kernel` 强制 `MATH` 或 `FLASH_ATTENTION`；
3. `reference_flash_attn`：如果安装了 Dao-AILab `flash-attn`，直接对齐官方实现。

每次 verify 输出里至少记录：

- input shape: `B, Hq, Hkv, Nq, Nkv, D`
- dtype: `fp16/bf16/fp32/fp8`
- causal/mask/dropout/GQA
- reference backend
- MAE / max error
- tolerance

如果 reference backend fallback 了，测试应该打印 warning，不能静默通过。

## 对 kernel 设计的影响

SDPA 告诉我们，attention kernel 的外部接口最好不要一开始就写死成“只有 Q/K/V/O”。至少应该预留：

- `is_causal`
- `attn_mask` 或 block mask metadata
- `softmax_scale`
- `dropout_p`，哪怕第一阶段不实现
- GQA/MQA 的 head mapping
- variable length / packed sequence 的可能入口

但是实现上不要一次全做。tiny-cutlass 当前更现实的节奏是：

1. dense non-causal, no dropout；
2. dense causal, no dropout；
3. GQA/MQA；
4. local/sliding window；
5. block mask 或 score modification；
6. dropout/backward。

每一步都要保持 reference parity，再看 benchmark。

## 当前结论

PyTorch 的 SDPA 路线是“统一语义入口 + backend 自动选择 + 可控 override”。这和我们做 tiny-cutlass 的方式刚好互补：我们写 kernel 时不要只盯单个 CUDA 文件，还要把 reference backend、dispatch 条件、fallback 和测试输出设计好。否则 kernel 可能已经很快，但根本不知道自己在和谁比。

## 资料来源

- PyTorch `scaled_dot_product_attention`: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- PyTorch `sdpa_kernel`: https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html
- PyTorch `SDPBackend`: https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.SDPBackend.html
- PyTorch FlexAttention blog: https://pytorch.org/blog/flexattention/
