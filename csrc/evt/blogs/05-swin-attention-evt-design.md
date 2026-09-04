# EVT 学习记（五）：SwinAttention 用 EVT 做融合的设计

## 前言

前四篇把 EVT 读完了：论文的抽象、两条实现路径、案例惯用法。这一篇回到最初的问题——
如果要用 EVT 融合 SwinAttention，设计应该长什么样。

**本篇只写设计，不写代码。** 目标是把「哪些能融、哪些不能、几种切法各自的代价」写清楚到
可以直接据此开工的程度，同时把不确定的地方标出来。

上下文：`csrc/swin` 工作区当前的 attention 路径复用
`csrc/flash-attention/02-tiled-online-attention` 的 flash kernel（
`csrc/swin/kernel/window_attention.h:77-85`），是 monolithic 单 kernel，**不是** EVT。
另有一个已验证的 EVT 实例只覆盖 QK^T 阶段（`csrc/swin/kernel/evt_attention_scores.h`，
MAE 7.66e-9）。所以这篇要回答的实际问题是：**EVT 在这条已有链路里应该占什么位置。**

## Overview

### 问题定义

Swin-T stage-1 的具体尺寸（`csrc/swin/swin_problem.h:16-25`）：

```text
batch=2, image=14, window=7, shift=3, heads=3, head_size=32
  → C = heads·head_size = 96
    L = window² = 49              （L_pad = 56，8 对齐）
    num_windows = (14/7)² = 4，BW = batch·num_windows = 8
    attention batch = BW·heads = 24
```

要算的数学：

```text
for each (window w, head h):
    S = scale · (Q · Kᵀ) + B_rel          Q,K ∈ R^{L×d},  S ∈ R^{L×L},  d = 32
    P = softmax_row(S)                     沿 key 轴（长度 L）归约
    O = P · V                              O ∈ R^{L×d}
然后 heads 拼回 C，走 proj GEMM
```

三个阶段：**GEMM → 行归约+逐元素 → GEMM**。

### 第一性结论：EVT 在这里的能力边界

把 `02`/`03`/`04` 篇的结论套到这个问题上：

| 阶段 | EVT 能做吗 | 依据 |
| --- | --- | --- |
| `scale · QKᵀ + B_rel` | ✅ 完全可以，已验证 | 纯 elementwise + scalar broadcast + aux load |
| mask（shifted window 的 attn_mask） | ✅ 同上，再加一个 aux load + plus | 同上 |
| `softmax_row` | ⚠️ 取决于 N-tile 能否覆盖整条 key 轴 | 见下 |
| `P · V` | ❌ epilogue 不能再发 matmul | `04` 篇末表 |

第三行是这篇的关键，之前的记录（memory `swin-evt-attention-boundary`）把它记成了「❌ 不能
做 softmax」。**这个结论需要修正为条件成立**，理由是 `03` 篇读到的两条事实：

1. Sm90 的 `reduce` 钩子拿到的 `visit_results` 是**可写的**寄存器 tensor
   （`sm90_epilogue_tma_warpspecialized.hpp:877` 把 `tRS_rCompute_frg` 传进去），
   `Sm90TopKSoftmaxColReduction` 就是靠它在 `reduce` 里原地改写、做出**完整** softmax 的
   （`sm90_visitor_topk_softmax.hpp:567` 起）。
2. 它成立的前提被源码写明了（`:716-718`）：「假设归约维上一行的所有元素都在同一个 epilogue
   subtile 内被访问」，加上 `static_assert(size<1>(warp_layout_MN) <= 1)`（N 方向只能一个 warp）。

而 Swin 的 `N = L = 49`。**一个 `ThreadblockShape::kN = 64` 的 tile 就完整覆盖了整条 key
轴**——`grid.n() == 1`。这正好落在那个前提里。所以：

> **在 Swin window attention 这个特定尺寸下（L ≤ kN），softmax 的归约维完整落在单个 CTA
> tile 内，EVT 原理上可以做完整 row-softmax。** 限制不在 EVT 抽象，在两点工程现实：
> (a) 2x 路径（Sm80/89）没有 `visit_results` 可写这个能力，也没有 row-softmax 节点；
> (b) 3x 路径有能力和 `reduce` 钩子，但现成的 `Sm90TopKSoftmax` 是 top-K 特化（`TopK ∈
> {2,4}`），不是通用 row-softmax，要新写一个节点。

这个修正很重要，因为它把「EVT 融不了 softmax」从一个原理性结论降级为一个**平台+库覆盖度**
结论。本仓库硬件是 SM89，所以实践上仍然是 (a) 生效——但理由要说对。

`P · V` 那一条是真正的原理性限制：epilogue 是 mainloop 之后的阶段，没有任何钩子能再启动一次
tensor-core MMA。要在一个 kernel 里做两次 GEMM 只有两条路——back-to-back（example 13，
accumulator 经 RF 或 smem 直接成为第二个 GEMM 的 A）或 monolithic 手写（example 41 /
flash attention）。**这两条都不是 EVT。**

### 三种候选设计

#### 方案 A：EVT 只做 scores，softmax 与 PV 各自成核（三核流水）

```text
kernel 1  batched GEMM Q·Kᵀ  +  EVT epilogue: scale·acc + B_rel + attn_mask  → S
kernel 2  row-softmax                                                        → P
kernel 3  batched GEMM P·V                                                   → O
```

这是当前已验证实例（`evt_attention_scores.h`）的自然延伸，只需给树再挂一个 mask 的
`VisitorAuxLoad` + `plus`：

```text
EVTStore = Sm80EVT<AuxStore(S),
             Sm80EVT<Compute(plus),
               Sm80EVT<Compute(homogeneous_multiply_add),
                 ScalarBroadcast(scale), AccFetch, AuxLoad(B_rel)>,
               AuxLoad(attn_mask)>>
```

- 优点：SM89 上今天就能做；EVT 部分已验证；每一段都能独立对 reference。
- 缺点：**S 和 P 都要走 DRAM**。`S` 的规模是 `BW·heads·L·L_pad = 24·49·56 ≈ 65.8K` 元素
  （fp16 约 132 KB），写一遍读一遍再写一遍读一遍。相比 flash 路径这是纯粹的退步。
- 定位：**它是教学/对照实例，不是生产路径。** 保持它存在的价值是它是本仓库唯一一个真 EVT
  核，用来定位 EVT 抽象本身的行为和 NCU 特征。

#### 方案 B：EVT 只吸收 scores 的 elementwise，softmax + PV 用 B2B 融合（两核）

```text
kernel 1  batched GEMM Q·Kᵀ  +  EVT epilogue: scale·acc + bias + mask       → S
kernel 2  fused (softmax → P·V)：softmax 与 PV 在一个 kernel 内经 smem 交接  → O
```

- 相比 A 省掉一次 `P` 的往返。
- 但 kernel 2 里 softmax 之后 P 要喂给 MMA，本质上就是 flash kernel 的后半段——写它的工作量
  接近写整个 flash kernel，却只拿到 flash 的一部分收益。
- 定位：**不推荐**。它处在「EVT 能力」和「monolithic 能力」之间的尴尬带，两边的优势都没吃满。

#### 方案 C（推荐）：monolithic flash kernel 做主体，EVT 用在 attention 的前后 GEMM 上

这是把 EVT 放在它真正有比较优势的位置。观察 SwinAttention 的完整数据流（
`csrc/swin/docs/00-overview.md:101-107`）：

```text
x → norm1 → shift → window_partition → QKV GEMM → [attention] → proj GEMM → window_reverse → +residual
                                        ↑EVT                      ↑EVT
```

attention 内部（QKᵀ/softmax/PV）交给 flash kernel——它已经验证过，且是这一段的正确技术。
EVT 用在**夹住 attention 的两个普通 GEMM 的 epilogue** 上：

**C1. QKV projection GEMM 的 epilogue。**

```text
mainloop: [BW·L, C] · [C, 3C]  → qkv
EVT 树：  AuxStore(Q) / AuxStore(K) / AuxStore(V) 三个输出
          + AccFetch + RowBroadcast(qkv_bias) + Compute(plus)
          + 可能的 layout 重排（de-interleave 到 per-(window,head) 连续）
```

关键点：**`VisitorAuxStore` 是透传节点**（`04` 篇），一棵树可以有多个 store。QKV 的三段
输出可以在同一个 epilogue 里写去三个不同的 buffer，用不同的 `StrideMNL` 表达布局。这直接
解决了当前 `evt_attention_scores.h` 头注释里说的那个问题——

> Swin's raw QKV is interleaved `[BW, L, C=heads*head_dim]`; a split/de-interleave step
> precedes this kernel

——那个 de-interleave pre-pass 可以被 QKV GEMM 的 EVT epilogue 吸收掉。这正是论文 5.2 节
Layout Node Elimination 的用途：reshape/permute 被推进 store 节点，变成 stride 的事。

需要确认的点：三个 store 的 N 范围是 `[0,C)`、`[C,2C)`、`[2C,3C)`，而 EVT 节点的 predicate
是 `elem_less(coord, problem_shape)` 的全局判断，没有「按列区间选 store」的机制。所以更可能
的形态是**三个独立的 GEMM**（各自 N=C），或者一个 GEMM 配 `Sm90SplitTreeVisitor` 式的多输出
——后者在 2x 上没有对应物。这是方案 C1 最大的未定项。

**C2. proj GEMM 的 epilogue。**

```text
mainloop: [BW·L, C] · [C, C]  → proj
EVT 树：  AuxStore(out) + Compute(plus)
                          ├─ Compute(plus): AccFetch + RowBroadcast(proj_bias)
                          └─ AuxLoad(residual)                     ← 残差直接吸进 epilogue
```

如果 window_reverse 能表达成 store 的 stride（window 坐标 → image 坐标的映射是仿射的吗？
见下面的遗留问题），那么 `proj + bias + residual + window_reverse` 是**一个 GEMM 一个
epilogue** 的事，省掉一次完整的 `[BW·L, C]` 往返。

- 优点：EVT 用在它最擅长的形状（GEMM + 逐元素 + 广播 + 多输出 + layout 折进 stride），
  attention 用在它最擅长的技术（monolithic online softmax）。两边都不勉强。
- 缺点：跨过了 EVT 与 flash kernel 的边界，Q/K/V 和 attention 输出仍要过 DRAM。
- 与 `csrc/swin` 既有方向的关系：`docs/00-overview.md:81-99` 的目标是「SwinBlock = 单
  per-window megakernel」，那比方案 C 更激进（连 DRAM 往返都不要）。方案 C 是**通往那个目标
  路上的中间形态**，也是在 megakernel 的 smem 预算被深层 stage（C=768）打爆时的退路。

### 推荐的落地顺序

```text
1. 保留方案 A 的 scores 核作为 EVT 对照实例（已完成，MAE 7.66e-9）
   → 给它补 NCU，作为"EVT epilogue 的访存/占用特征"基线
2. 方案 C2（proj GEMM + bias + residual EVT epilogue）
   → 最小、最独立、收益明确；window_reverse 是否能进 stride 单独验证
3. 方案 C1（QKV GEMM epilogue 吸收 bias + de-interleave）
   → 先确认多输出/分段 store 在 2x 上的可行形态
4. 方案 B 直接跳过
5. 长期：per-window megakernel（docs/00-overview.md §3.2），EVT 退化为它的内部 epilogue 写法
```

每一步的验证 gate 沿用仓库规则：`build → verify → bench`，未过 reference parity 不 benchmark。

### 一个明确的非目标：不要试图用 EVT 融合完整 attention

写下来免得以后再绕：

- `P · V` 进不了 epilogue（原理性，无解法）；
- 即使在 Sm90 上新写一个 row-softmax EVT 节点、即使 `L ≤ kN` 让它数学上成立，融合出来的
  也只是 `QKᵀ + softmax`，PV 仍要第二个核；
- 而 `QKᵀ + softmax + PV` 一个核就是 flash attention，本仓库已经有了
  （`csrc/flash-attention/02-tiled-online-attention`，已验证）。

**EVT 与 flash attention 不是竞争关系，是互补关系**：flash 处理「两个 GEMM 夹一个跨轴归约」
这个特殊结构，EVT 处理「一个 GEMM 后面挂一串逐元素/广播/多输出/layout 变换」这个通用结构。
Swin block 里两种结构都有。

### 如果一定要在 Sm90 上试通用 row-softmax 节点

留个设计草稿，因为它是唯一能扩展 EVT 能力边界的方向（本机 SM89 跑不了，仅供记录）：

```text
节点：Sm90RowSoftmax（新写，仿 Sm90TopKSoftmaxColReduction 的结构）
  visit(frg_acc, epi_v, epi_m, epi_n, frg_input):
      累积 running max 到寄存器 tensor（不改数据，透传 frg_input）
  reduce(smem_buffer, sync_fn, epi_m, epi_n, is_last_iteration, visit_results):
      1. 跨 lane butterfly reduce max
      2. 重访 visit_results：exp(x - max)，同时累积 sum
      3. 跨 lane butterfly reduce sum
      4. 再重访 visit_results：除以 sum
  前提断言（照抄 Sm90TopKSoftmax）：
      static_assert(size<1>(warp_layout_MN) <= 1)   // N 方向单 warp
      + 运行期要求 problem N ≤ EpilogueTile N
```

第 2、4 步要两遍重访 `visit_results`，`Sm90TopKSoftmax` 只需一遍（因为 logsumexp 技巧把
max 和 sum 合并成一次归约）。同样的技巧可以用在这里：跟踪 `logsumexp` 而不是分别跟踪 max
和 sum，就能压回一遍重访——`sm90_visitor_topk_softmax.hpp:345-355` 的注释把这个推导写得很
清楚，值得直接借。

但要清楚这个节点的收益：它能把 `scale·QKᵀ + bias + mask + softmax` 融成一个核，省掉 `S`
的一次往返。相比 flash 仍然多一次 `P` 的往返 + 一次 PV 核启动。**只有在「已经有一个高质量
的 batched GEMM mainloop、且不想写 monolithic kernel」时才划算。**

## 遗留问题（做之前必须先确认）

1. **`window_reverse` 能否表达成 store 的 stride？** window 坐标 →image 坐标是
   `(w_idx, token_idx) → (b, h, w)` 的分块置换。cute stride 表达的是仿射映射，而分块置换在
   `H/window` 整除时**可能**可以用多维 stride 表达（形如
   `(bw, l, c) → ((bw/nw), (l/ws), (l%ws), c)` 的 layout 重解释）。要先在纸上把 layout 写出来
   验证，不能假设可以。shift 的循环移位更可疑（它是模运算，不是仿射）。
2. **2x 路径上「一棵树多个 store 写不同列区间」有没有可行形态？** `VisitorAuxStore` 的
   predicate 是全局 `elem_less(coord, problem_shape)`，没有列区间选择。可能的替代是
   `Sm80TopologicalVisitor` + 每个 store 用不同 `StrideMNL` 把同一 tile 映射到不同 buffer，
   但这仍然是「所有 store 都写全 tile」，不是分段。需要实验。
3. **QKV GEMM 用一个 N=3C 的 GEMM 还是三个 N=C 的 GEMM？** 前者 mainloop 更高效（一次读
   activation），后者 epilogue 简单。要看 `[BW·L=392, C=96]` 这个规模下哪边占优——`392×288`
   的输出 tile 在 SM89 上能开多少 CTA，是不是已经 SM 不足。
4. **conv 侧无 EVT 入口**（`02` 篇遗留问题）：`include/cutlass/conv/` grep 不到
   `EpilogueWithVisitor`/`FusionCallbacks`。PatchEmbed 那条线（`docs/00-overview.md:69-76`）
   要 fork conv kernel，与本篇的 GEMM 路径不共享 EVT 装配。
5. **flash kernel 的 bias 已经在 fragment 级加了**（`kernel/window_attention.h:30-32`），
   方案 C 下 `B_rel` 不需要 EVT 参与；只有方案 A/B 才需要。这点在选方案时别重复计算收益。
6. **memory 里的旧结论要更新**：`swin-evt-attention-boundary` 记的是「EVT 不能做 softmax」，
   应改为「2x 路径不能；3x 在归约维落单 warp 时可以，但库里只有 top-K 特化」。

## Profile

<!-- 设计篇，无实测。方案 A 的 scores 核 NCU 数据归属 csrc/swin，待 bench 通过后补。 -->

## 后记

<!-- HUMAN: 选定方案后补充自己的判断、踩坑和下一步。 -->
