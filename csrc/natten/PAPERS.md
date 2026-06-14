# NATTEN 论文导读

这份导读面向 tiny-cutlass 的实现工作，不是纯模型综述。阅读重点是：

- Neighborhood Attention 和 Swin Window Attention 的结构差异。
- Fused Neighborhood Attention 为什么需要专门 kernel。
- 正确性验证应该覆盖哪些语义。
- 性能瓶颈可能来自哪里，以及后续 CUTLASS 实现应怎样落点。

## 建议阅读顺序

1. Swin Transformer：先理解固定窗口 + 平移窗口这条对照线。
2. Neighborhood Attention Transformer (NAT)：理解滑动邻域 attention
   为什么比 Swin 更接近卷积式局部归纳偏置。
3. Dilated Neighborhood Attention Transformer (DiNAT)：理解 dilation 怎样扩大感受野，
   以及为什么问题描述符里必须保留 `dilation`。
4. Faster Neighborhood Attention：重点读 kernel/系统部分，它直接解释 FNA 为什么要融合。
5. Generalized Neighborhood Attention：理解后续更通用的稀疏 attention/NA 统一方向。

## 论文速览

| 论文 | 资料 | 核心问题 | 对 tiny-cutlass 的实现启发 |
| --- | --- | --- | --- |
| Swin Transformer: Hierarchical Vision Transformer using Shifted Windows | <https://arxiv.org/abs/2103.14030> | 用不重叠窗口限制 attention 复杂度，再通过平移窗口让窗口之间交换信息。 | Swin 的窗口 token 是连续块，比较容易复用 GEMM/Fused Attention；这是 NATTEN 后续要对齐代码风格的基线。 |
| Neighborhood Attention Transformer | <https://arxiv.org/abs/2204.07143> | 每个 token 关注自己附近的滑动邻域，保留局部归纳偏置和平移等变性。 | FNA 的核心不是普通固定窗口 attention；每个 query 的 K/V neighborhood 会滑动，边界和索引语义必须进 reference parity。 |
| Dilated Neighborhood Attention Transformer | <https://arxiv.org/abs/2209.15001> | 在 neighborhood attention 上引入 dilation，用稀疏采样扩大感受野。 | `FnaForwardProblem` 需要保留 `dilation`；测试要覆盖 `dilation > 1` 的索引和边界。 |
| Faster Neighborhood Attention | <https://arxiv.org/abs/2403.04690> | 朴素 NA kernel 很容易被访存、索引和非规则局部窗口拖慢；论文围绕 fused neighborhood attention 做系统优化。 | 后续真实实现应走 fused `QK -> softmax -> PV`，避免落全局 score/probability 矩阵；性能测试必须在 reference parity 后进行。 |
| Generalized Neighborhood Attention | <https://arxiv.org/abs/2504.16922> | 把 neighborhood attention 推到更通用的稀疏/块稀疏 attention 表达，并强调高性能 kernel 生成和调度。 | 后续 policy 不应写死 1D/SM80/单一 tile；rank、arch、dtype、tile、mask 都应在模板 policy 或显式 descriptor 里表达。 |

## NAT 和 Swin 的关键差异

Swin 的 attention 单元是固定窗口。先把图像切成 `[B * num_windows, L, C]`，再在每个窗口内做 dense attention。平移窗口通过移动窗口边界引入跨窗口交互。

NAT 的 attention 单元是每个 query token 的滑动邻域。相邻 query 的 neighborhood 大量重叠，但并不是同一个固定窗口。这让模型语义更像卷积：局部、平移等变、邻域中心随 query 移动。

这两个差异直接影响 kernel：

- Swin 更容易把一个窗口当作小型 dense attention，layout 通常更规整。
- NAT/FNA 的 K/V 访问带有 per-query neighborhood 索引，边界、padding、dilation、causal mask 都更容易引入 predicate 和非连续访存。
- Swin 的 shifted-window mask 是 dense bias/mask；NAT 的 mask 更像局部邻域有效性判定。
- 统一代码风格时，可以统一 problem/test/script/document 组织，但不能把 NAT 误写成 Swin window attention。

## Neighborhood 在哪里取

NAT 里的 neighborhood 取在空间 token 网格上，不取在 channel/head_dim 维度上。
以一张 `128 x 128` 图像为例，如果 patch embedding 的 patch size 是 `8 x 8`，
则空间上会得到 `16 x 16` 个 token。每个 token 可以是一个 `C = 64` 维向量：

```text
image:      128 x 128
patch:        8 x   8
token grid:  16 x  16
token dim:   C = 64
```

线性投影后，每个空间位置都有自己的 `Q[row, col]`、`K[row, col]` 和
`V[row, col]`。如果再拆成多头，那么单个 head 里的 `Q[row, col, head]`
是一个 `head_dim` 维向量。attention 的点积发生在这个 `head_dim` 向量上，
但邻居选择发生在 `16 x 16` 这个空间网格上。

例如 `kernel_size = (3, 3)`、`dilation = (1, 1)` 时，中心 query 的邻居是：

```text
token grid around query (8, 8):

.  .  .  .  .
.  K  K  K  .
.  K  Q  K  .
.  K  K  K  .
.  .  .  .  .

neighbors =
  (7,7), (7,8), (7,9)
  (8,7), (8,8), (8,9)
  (9,7), (9,8), (9,9)
```

相邻 query 的 neighborhood 会一起滑动，而不是共享同一个固定窗口：

```text
Q(4,4) -> rows 3..5, cols 3..5
Q(4,5) -> rows 3..5, cols 4..6
Q(4,6) -> rows 3..5, cols 5..7
```

边界位置也不是在越界位置补零。以左上角 query 为例，窗口会向图像内部对齐：

```text
Q(0,0) -> rows 0..2, cols 0..2

Q  K  K  .  .
K  K  K  .  .
K  K  K  .  .
.  .  .  .  .
.  .  .  .  .
```

因此实现 reference 时要区分三件事：

- `16 x 16` 是 token 的空间分布。
- `C = 64` 或 `head_dim` 是每个 token 向量的通道维度。
- neighborhood 在空间分布里找相邻 token，之后才用对应 token 的向量做
  `QK -> softmax -> PV`。

## 对正确性的要求

当前仓库还没有真实 FNA kernel，所以只能做接口冒烟测试，不能宣称数值正确。后续补 kernel 时，verify 至少要覆盖：

- 1D non-causal FNA 的基础路径。
- `kernel_size` 为奇数和边界 query 的 neighborhood 向内对齐。
- `dilation = 1` 和 `dilation > 1`。
- `head_dim` 与 `head_dim_value` 不同的情况。
- `scale` 对 softmax 的影响。
- 输出 `O` 和需要的话 `logsumexp`。
- 小 shape 的 host/PyTorch/NATTEN reference parity。

推荐先从非常小的 1D case 开始，例如：

```text
B = 1
L = 8 或 16
heads = 1
head_dim = 16 或 32
kernel_size = 3 或 5
dilation = 1 或 2
```

只有 reference parity 通过后，`scripts\kernels\natten\natten.bat` 才能进入 benchmark。

## 对性能测试的要求

性能测试必须跟 `cutlass-kernel` skill 保持一致：

- build output 和 profiling artifact 只能落在 `build/`。
- `.bat` 顺序必须是 build -> verify -> bench。
- verify 失败时不能继续 bench。
- benchmark 结果必须说明 reference、shape、dtype、GPU、编译 arch、误差阈值。
- Nsight Compute/Systems 的 `.ncu-rep`、`.nsys-rep` 和 CSV 应放在
  `build/reports/natten`。

当前没有真实 runnable kernel，因此不能做性能结论。现在能做的只有：

- build + 接口冒烟测试。
- 从论文和代码结构推导可能瓶颈。
- 规划后续 reference parity 和 benchmark harness。

## 预期性能瓶颈

以下是后续实现时最需要盯的瓶颈假设：

- 非 fused 路径会把 attention score/probability 写回全局内存，通常会被带宽和中间矩阵大小拖住。
- sliding neighborhood 导致相邻 query 重复读取大量 K/V，cache/shared memory reuse 设计会决定上限。
- 边界 query、causal mask、dilation 会引入 predicate，处理不好会造成 warp 内分歧和无效 TensorOp work。
- `kernel_size`、`head_dim`、tile shape 不匹配时，Tensor Core 利用率可能低于 Swin 的固定 dense window attention。
- `dilation > 1` 会让 K/V 访问更不连续，load coalescing 更难。
- softmax 的 max/sum/logsumexp 需要寄存器和 shared memory 协同，过大的 tile 可能提高 occupancy 压力。
- 如果为了支持更多 shape 悄悄退回 SIMT/raw CUDA fallback，会破坏本工作区的目标；不支持的 shape 应显式拒绝。

## 对后续 CUTLASS 实现的落点

后续添加真实 kernel 时，优先保持下面的分层：

- `DefaultFnaForwardPolicy<...>` 继续作为模板 policy 工厂。
- `FnaForwardProblem` 承载 rank 相关 problem metadata；2D/3D 不要靠新增一堆 concrete 函数名表达。
- device/launcher 层负责把 policy 映射到底层 CUTLASS TensorOp/FMHA 实例。
- reference test 放在 `csrc/tests/natten`。
- benchmark/profiling 由 `scripts\kernels\natten\natten.bat` 驱动。

这样 NATTEN 才能和 Swin 工作区统一：文档、脚本、测试组织统一；kernel 语义和 policy 仍然保留 NAT/FNA 自己的差异。
