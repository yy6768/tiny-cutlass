# CUTLASS学习记（八）：Flash Attention Forward — 融合算子

## 前言

- 紧接上一部分 online softmax，发现没有明显的性能提升。这篇沿着原论文第三章，结合 example 41 写出 fused FlashAttention forward。
- 参考资料：
  - [[2205.14135\] FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) 原论文第三节
  - [Attention优化📚原理篇: 从Online-Softmax到FlashAttention V1/V2/V3 - 知乎](https://zhuanlan.zhihu.com/p/668888063) 的 FlashAttention v1/v2 节
  - [UWCSE 599M Note——From Online Softmax to FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf) 的最后一节

## Tiling

对于 forward 来说，FlashAttention v1 的核心就是 tiling。

一个 CTA 把 $Q$、$K$、$V$ 都切成 tile，然后在一个 kernel 内部流式处理：

- 每个 CTA 负责一块 $Q_i$ tile，行数为 $B_r$
- 沿 $K/V$ 方向循环，每次取一块 $K_j$ 和对应的 $V_j$，行数为 $B_c$
- 当前 tile 的 score $S_{ij}$ 算完后，在 register 里做 online softmax，得到 $P_{ij}$
- $P_{ij}$ 通过 shared memory 交给 MM1，和 $V_j$ 做矩阵乘并累加到 $O_i$

数据流变成：

$$
\text{for each } K_j\text{ tile}: \quad S_{ij} = Q_i K_j^T \xrightarrow{\text{register}} P_{ij} \xrightarrow{\text{smem}} O_i \mathrel{+}= P_{ij} V_j
$$

这里有两层不同粒度的 tiling：

- **CTA 级 tiling**：grid 静态分配 $Q_i$，一个 CTA 生命周期内 $i$ 不变。
- **K/V 流式 tiling**：同一个 CTA 在主循环中依次扫描 $K_j$、$V_j$，$j$ 每轮变化。

后一层才是 IO-aware 的核心：$S$、$P$ 只在当前 tile 内存在，不再完整 materialize 到 HBM。

## 实现

02 基本复现 example 41 的 forward 路径。下面按照一次 kernel 的实际执行顺序，从 launch 走到最终写回。

```text
tiled_online_attention.cu  -> 选择 policy 并启动 kernel
attention_kernel_batched_impl
  -> advance_to_block
  -> attention_kernel
     -> MM0
     -> iterative_softmax
     -> accumToSmem
     -> MM1
     -> epilogue
```

### Kernel launch

`tiled_online_attention.cu` 先根据 problem 选择 `TiledOnlineAttentionConfig`。`launch_attention` 填好 `Attention::Params`，检查 workspace、dynamic shared memory 和当前 shape 是否支持，然后启动：

```cpp
constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
kernel_fn<<<params.getBlocksGrid(), params.getThreadsGrid(), smem_bytes, stream>>>(params);
```

grid 是：

```cpp
dim3(
    ceil_div(num_queries, kQueriesPerBlock),
    num_heads,
    num_batches)
```

因此 `blockIdx.x` 选择 Q tile，`blockIdx.y` 选择 head，`blockIdx.z` 选择 batch。threadblock 内部则由 `kNumWarpsPerBlock` 个 warp 协作完成两次 GEMM 和 softmax。

kernel 入口：

```cpp
if (!p.advance_to_block()) {
  return;
}
AK::attention_kernel(p);
```

### `advance_to_block`

`advance_to_block` 把 Q、K、V、O 的基址移动到当前 CTA 负责的位置：

```cpp
auto batch_id = blockIdx.z;
auto head_id = blockIdx.y;
auto query_start = blockIdx.x * kQueriesPerBlock;

query_ptr += (q_start + query_start) * q_strideM + head_id * q_strideH;
key_ptr += k_start * k_strideM + head_id * k_strideH;
value_ptr += k_start * v_strideM + head_id * v_strideH;
output_ptr +=
    int64_t(q_start + query_start) * o_strideM + head_id * head_dim_value;
```

Q 指针直接移动到 $Q_i$，因为这个 CTA 只负责这一块 query。K 和 V 指针只移动到当前 batch 和 head 的开头，没有加 key tile 偏移。后面的 `iter_key_start` 循环会让同一个 CTA 扫过所有 $K_j/V_j$。

进入 `attention_kernel` 后，先初始化每一行的 online softmax 状态：

```text
m_prime = -inf
mi = -inf
s_prime = 0
out_rescale = 1
accum_o = 0
```

然后沿 key 方向循环：

```cpp
for (int32_t iter_key_start = 0;
     iter_key_start < p.num_keys;
     iter_key_start += kKeysPerBlock) {
  ...
}
```

这个循环体就是一个完整的 FlashAttention tile。

### MM0：计算 $Q_iK_j^T$

MM0 仍是 CUTLASS threadblock-scoped GEMM：Q/K 通过 iterator 进入 shared memory，再由 warp MMA 产生 score accumulator。02 的差异不在于换掉 GEMM，而在于 accumulator 后续不走普通 global-memory epilogue：它先进入 tile 内的 online softmax，再由 `B2bGemm::accumToSmem` 交给 MM1。

相关的局部组件是：

- `MakeCustomMma` 按 `kSingleValueIteration` 选择 MMA 路径；
- `AccumLambdaIterator` 按 attention score 的逻辑行解释分散在 warp/lane 中的 accumulator；
- `BiasLoader` 仅在启用 bias 的配置中参与线性融合；
- `B2bGemm` 提供 `accumToSmem`，把 softmax 后的当前 tile 写进 MM1 可读的 shared-memory layout。

具体数据地址如下：

```cpp
typename MM0::IteratorA iterator_A(..., p.query_ptr, ...);
typename MM0::IteratorB iterator_B(
    ...,
    p.key_ptr + iter_key_start * p.k_strideM,
    ...);
```

- Q 的地址在 CTA 开始时已经固定为 `query_ptr`。
- K 的地址随 `iter_key_start` 前进。`MM0::Mma` 使用 CUTLASS 的 threadblock-scoped GEMM 主循环，将 Q、K tile 搬到 shared memory，再由 warp MMA 计算：

```cpp
typename MM0::Mma::FragmentC accum;
accum.clear();
mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);
```

此时 `accum` 是 $S_{ij}=Q_iK_j^T$，仍然保留在各线程的 register fragment 中。MM0 没有把它通过普通 GEMM epilogue 写回 global memory，而是直接进入 online softmax。

### `iterative_softmax`

`iterative_softmax` 接收 MM0 留在 register fragment 中的 `accum`，对当前 score tile 做行归约，并更新跨 tile 保存的状态。shared memory 要等到 softmax 完成后，才由 `accumToSmem` 接收当前的 $P_{ij}$。

第 $j$ 轮的递推是：

$$
m_i^{new}=\max\left(m_i^{old},\operatorname{rowmax}(S_{ij})\right)
$$

$$
r_i=\exp\left(m_i^{old}-m_i^{new}\right)
$$

$$
P_{ij}=\exp\left(S_{ij}-m_i^{new}\right)
$$

`AccumLambdaIterator` 根据 accumulator layout 找到每个元素所属的行，先归约 row max，再计算指数和 row sum。函数结束时，`accum` 已经从 $S_{ij}$ 原地变成未归一化的 $P_{ij}$。

旧输出也必须乘 `out_rescale`。如果新 tile 提高了行最大值，前面 tile 累积的分子和分母都要换到新的指数基准。否则不同 tile 的结果不能直接相加。

### `accumToSmem`：连接两次 GEMM

softmax 之后调用：

```cpp
MM0::B2bGemm::accumToSmem(
    shared_storage.after_mm0.si,
    accum,
    my_lane_id,
    output_tile_coords);
```

代码的逻辑里做了两件事

- `FragmentIteratorAccumulator` 将 Tensor Core accumulator 中的 $P_{ij}$ 按逻辑顺序拆成片段并写入 shared memory；
- `SmemIteratorD0` 再把这些片段写入 row-major shared memory。MM1 使用与这个格式配套的 custom warp iterator 读取。

所以两次 GEMM 的交接是：

```text
MM0: Q_i K_j^T -> S_ij in registers
online softmax: S_ij -> P_ij
accumToSmem: P_ij -> shared memory
MM1: shared-memory P_ij @ V_j
```

`P_{ij}` 只在当前 CTA 内部短暂存在，不会形成完整的 global-memory 矩阵。这一步是 02 和 01 真正的分界。

### MM1：计算 $P_{ij}V_j$

MM1 的 operand A 来自 `shared_storage.after_mm0.si`，operand B 是当前的 V tile：

```cpp
typename MM1::Mma::IteratorB iterator_V(
    ...,
    p.value_ptr + iter_key_start * p.v_strideM,
    ...);

typename MM1::Mma mma_pv(
    shared_storage.after_mm0.si.accum_ref(),
    shared_storage.after_mm0.mm1.operand_B_ref(),
    thread_id(),
    my_warp_id,
    my_lane_id);

mma_pv(gemm_k_iterations, accum_o, iterator_V, accum_o);
```

GEMM 计算 $P_{ij}V_j$。`accum_o` 保存当前 Q tile 的输出分子，并在 key tile 之间继续累积。

当 `kKeepOutputInRF` 为真，`accum_o` 一直留在寄存器里。

每轮 softmax 更新最大值后，`iterative_softmax` 直接用 `out_rescale` 重缩放旧的 `accum_o`，然后 MM1 累加当前 tile 的贡献。

当输出不能一直保留在 register 时，每轮 MM1 后通过 epilogue 把中间结果写到 `output_accum_ptr`，下一轮再读回来继续合并。两条路径的 online softmax 逻辑相同，区别只是输出分子的保存位置。

### Epilogue：合并与归一化

MM1 使用 `MemoryEfficientAttentionNormalize` 处理当前 accumulator、旧输出和 online softmax 状态。核心关系是：

```cpp
alpha = isLast ? 1 / s_prime[row] : 1;
beta = alpha * out_rescale[row];
output = alpha * accumulator + beta * source;
```

中间 key tile 只合并新旧输出，不做最终除法。最后一个 key tile 才使用 `1 / s_prime` 完成 softmax 归一化，并写回 $O_i$。

如果 `kKeepOutputInRF` 为真，key 循环结束后只执行一次最终 epilogue。如果为假，每轮都通过 epilogue 保存中间输出，最后一轮切换到最终输出 iterator。

到这里一次完整执行过程才结束：launcher 决定 tile policy，grid 分配 $Q_i$，CTA 循环读取 $K_j/V_j$，MM0 产生 score，online softmax 更新状态，MM1 累积输出，最后 epilogue 归一化并写回。

## 后记

- 本篇结合 example 41 把 FlashAttention forward 的数据路径走了一遍：完整 $P$ 不再落到 global memory，但当前 tile 仍通过 shared memory 从 MM0 交给 MM1。
- 02 具备沿 Q 划分 CTA、延迟最终归一化等结构；它不等同于 FA2，warp ownership 和 register-sourced MM1 仍是下一步。
- 本轮只整理代码路径。性能结论要等 02 重新通过 cuDNN reference，并把对应的 benchmark、NCU CSV 和 `.ncu-rep` 一起落到 `build/` 后再补。
