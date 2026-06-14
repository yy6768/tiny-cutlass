# Attention 量化融合设计：INT8 / FP8 应该怎么接进网络

## 前言

这里先不急着写代码。我想把问题拆清楚：如果 attention 本身已经融合了 `QK^T -> softmax -> PV`，那量化节点到底应该放在哪里？是把 Q/K/V 先量化好再进 kernel，还是在 kernel 里顺手 quantize/dequantize，还是像 Transformer Engine 那样把 scale metadata 变成整个网络的一等公民？

我的初步判断是：INT8 和 FP8 不应该用同一套设计。INT8 更像 inference-oriented 的显式量化数据路径；FP8 更像训练/推理都可能用的低精度 compute path，需要 scale recipe、amax、outlier 处理和 kernel pipeline 一起设计。

## 先定义边界

我们要融合的不是一个孤立 quantize kernel，而是一个小网络片段：

```text
X
 -> QKV projection
 -> optional RoPE / bias / norm
 -> attention(Q, K, V)
 -> output projection
 -> residual / norm / MLP
```

如果只在 attention kernel 入口做 `Q_fp16 -> Q_int8 -> dequant -> MMA`，通常意义不大，因为你多了一次量化和反量化，却没有减少上游/下游 memory traffic。真正值得融合的是：

- projection 输出直接写成 quantized Q/K/V；
- attention kernel 直接消费 quantized Q/K/V 和 scale；
- attention 输出直接给下一层 GEMM 消费，必要时写成 quantized O；
- scale/amax metadata 沿网络传递，而不是每个节点临时重算。

## INT8 路线

INT8 更适合 inference。它的工程重点是 scale granularity 和 calibration。

### 数据格式

可选粒度：

- per-tensor scale：实现简单，但 outlier 容易毁掉精度；
- per-channel/per-head scale：适合 Q/K/V projection 输出；
- per-block scale：更适合 attention tile，但 metadata 更多；
- per-token dynamic scale：精度好，但 scale 计算和写回成本高。

对 attention 来说，我更倾向从 per-head 或 per-block 开始。per-tensor 太粗，per-token 太容易把 kernel 复杂度拉爆。

### kernel 输入

一个 INT8 attention kernel 至少需要：

```text
Q_int8, K_int8, V_int8
scale_q, scale_k, scale_v
scale_softmax 或 softmax_scale
O_dtype / O_scale
```

`QK^T` 的 scale 是 `scale_q * scale_k * softmax_scale`。softmax 本身仍建议在 FP32 或至少 FP32 accumulator 路径里做，因为 exp/sum 对量化误差非常敏感。

### 融合点

INT8 第一阶段可以这样设计：

1. QKV projection GEMM 输出 INT8 Q/K/V，同时输出 scale；
2. attention kernel 读取 INT8 Q/K/V，dequant 到 MMA fragment 或 accumulator 路径；
3. softmax 使用 FP32 状态；
4. `P @ V` 后输出 FP16/BF16；
5. 第二阶段再尝试把 O quantize 成 INT8，直接喂给 output projection。

这里不要一开始就追全链路 INT8。先把 Q/K/V INT8 输入做对，再考虑 O INT8 输出。

## FP8 路线

FP8 更贴近 Hopper/Blackwell 的 Tensor Core 路线。FlashAttention-3 和 NVIDIA Transformer Engine 都把 FP8 当成需要 recipe 的系统，而不是单个 dtype。

常见 FP8 格式包括 `E4M3` 和 `E5M2`。粗略地说：

- `E4M3` mantissa 多一点，适合 activation/weight；
- `E5M2` exponent 多一点，适合 gradient 或动态范围更大的数据；
- 实际选择要看硬件、训练/推理、scale 策略和 reference error。

Transformer Engine 里已经有 FP8 autocast、FP8/MXFP8/block scaling、FP8 tensor/quantizer、`DotProductAttention`、`MultiheadAttention` 和 operation fuser 这些抽象。我们不一定直接依赖 TE，但它提示了一件事：FP8 的 scale 是系统级状态。

## FP8 attention 的设计草图

### 元数据

FP8 path 至少需要这些 metadata：

```text
Q_fp8, K_fp8, V_fp8
scale_q, scale_k, scale_v
amax_q, amax_k, amax_v
scale_inv_q, scale_inv_k, scale_inv_v
scale_o / amax_o
recipe: current / delayed / block / MXFP8
```

如果使用 block scaling，还要定义 scale block shape，例如按 head_dim block、token block，或者和 MMA tile 对齐。

### Kernel 内部

attention kernel 内部可以按这个路径走：

1. load FP8 Q/K tile；
2. apply scale inverse，进入 WGMMA/MMA；
3. QK accumulator 使用 FP32；
4. online softmax 状态 `m_i/l_i` 使用 FP32；
5. load FP8 V tile；
6. `P @ V` accumulator 使用 FP32 或 FP16/F32 policy；
7. 输出 FP16/BF16，或根据下一层需要直接 quantize 成 FP8。

如果目标是 SM90，WGMMA FP8 路线应该单独设计，不要强行塞进 SM80 kernel。

## Outlier 和 Hadamard / incoherent processing

FA3 的一个关键提醒是：activation outlier 会让 FP8 误差明显变差。它使用 random-sign Hadamard transform 做 incoherent processing，把 outlier 能量摊开。这个变换是 memory-bandwidth bound，所以可以和 RoPE 这类 memory-bound 操作融合。

对我们来说，一个合理的实验路线是：

1. baseline FP8：只做 scale/dequant，没有 outlier 处理；
2. FP8 + per-head/per-block scale；
3. FP8 + Hadamard transform；
4. FP8 + Hadamard fused with RoPE；
5. 对比 MAE/max error 和 Nsight profile。

这条线必须把 numerical validation 当成第一目标。FP8 kernel 快但 error 爆掉，没有意义。

## 网络级融合设计

我更倾向把量化设计成 graph-level contract：

```text
Linear(QKV)
  output: Q_fp8/K_fp8/V_fp8 + scales + amax
RoPE/Hadamard
  input/output: quantized tensor + updated scale
Attention
  input: quantized Q/K/V + scale metadata
  output: fp16/bf16 or quantized O
Linear(Output)
  consume: O or O_fp8 + scale
```

这样每个节点都知道自己消费什么 dtype、产生什么 metadata。kernel fusion 时可以把相邻 memory-bound 节点合并，但语义上不丢 scale。

## tiny-cutlass 的实现建议

### 阶段 0：reference

先写纯 PyTorch reference：

- FP16/BF16 attention；
- simulated INT8 quant/dequant；
- simulated FP8 quant/dequant；
- optional Hadamard；
- 输出 MAE/max error。

没有 reference，就不要写 CUDA。

### 阶段 1：INT8 Q/K/V 输入

写一个只消费 INT8 Q/K/V、输出 FP16/BF16 的 attention variant。先不量化 O。目标是验证 scale 公式和 softmax 数值路径。

### 阶段 2：FP8 Q/K/V 输入

SM90 优先，使用 CUTLASS/CuTe FP8 MMA/WGMMA。输出先保持 FP16/BF16。目标是把 FP8 Tensor Core 路径跑通。

### 阶段 3：O 量化

当下一层 output projection 也能消费 quantized O 时，再把 O quantize 融进 epilogue。否则它只是多做一步。

### 阶段 4：RoPE/Hadamard 融合

把 RoPE 和 Hadamard 作为 memory-bound transform 融到 Q/K load 或 QKV projection epilogue。这里要小心：如果变换改变了 layout，attention kernel 的 shared memory swizzle 也要重新审查。

## Profiling 指标

INT8/FP8 不能只看 Tensor Core 利用率。至少要记录：

- `launch__registers_per_thread`
- `launch__shared_mem_per_block`
- `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`
- `sm__issue_active.avg.pct_of_peak_sustained_elapsed`
- `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`
- scale/amax kernel 的时间，或者 fused 后的额外指令成本
- Nsight Systems 里 quantize/attention/projection 是否真的减少 kernel launch 和 memory round-trip

## 当前结论

量化融合的核心不是“把 quantize 函数塞进 attention kernel”。真正的问题是让 dtype、scale、amax、outlier 处理和下游消费关系变成网络级 contract。INT8 可以先从 inference 的 Q/K/V quantized input 做起；FP8 应该按 SM90/WGMMA/Transformer Engine/FA3 的思路单独推进。每一步都先过 reference parity，再谈性能。

## 资料来源

- FlashAttention-3 paper: https://tridao.me/publications/flash3/flash3.pdf
- FlashAttention-3 official blog: https://tridao.me/blog/2024/flash3/
- NVIDIA Transformer Engine docs: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html
- PyTorch FlexAttention blog: https://pytorch.org/blog/flexattention/
