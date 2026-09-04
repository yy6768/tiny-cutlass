# EVT 学习记（四）：结合案例读 EVT

## 前言

`02`/`03` 篇把两条实现路径读完了。这一篇反过来，从具体案例回看抽象：一棵树在真实
example 里长什么样、`Arguments` 的嵌套顺序为什么那么反直觉、EVT 用不了的时候 CUTLASS 是
怎么写的（example 35/37），以及本仓库那个已经跑通 reference parity 的 EVT 实例
（`csrc/swin/kernel/evt_attention_scores.h`）逐层是怎么拼起来的。

案例清单：

| 案例 | 路径 | 看什么 |
| --- | --- | --- |
| example 47 | `examples/47_ampere_gemm_universal_streamk/ampere_gemm_universal_streamk_broadcast.cu` | 最完整的 Sm80EVT 惯用法（4 层树 + StreamK） |
| example 15 | `examples/15_ampere_sparse_tensorop_gemm/ampere_sparse_tensorop_gemm_with_visitor.cu` | 最小的 Sm80EVT + 稀疏 mainloop |
| example 49 | `examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu` | Sm90 CustomEVT 与 builder 的关系 |
| example 35 | `examples/35_gemm_softmax/` | EVT 之前的手写 visitor；GEMM+softmax 为什么是 3 个 kernel |
| example 37 | `examples/37_gemm_layernorm_gemm_fusion/` | 手写 visitor 做 partial LayerNorm 归约 |
| 本仓库 | `csrc/swin/kernel/evt_attention_scores.h` | 已验证的 batched GEMM + EVT（MAE 7.66e-9） |

## Overview

### 案例一：example 47 —— Sm80EVT 的标准惯用法

要算的东西：`D = relu?(acc + bias_row) + C1 + C2`（实际是三次 `plus`，没有激活）。树的
搭法（`ampere_gemm_universal_streamk_broadcast.cu:177-239`）：

```text
EVTD = Sm80EVT<D,
         Sm80EVT<Compute2,                                   // + C2
           Sm80EVT<Compute1,                                 // + C1
             Sm80EVT<Compute0,                               // acc + bias
               Accum,                                        //   VisitorAccFetch
               Bias>,                                        //   VisitorRowBroadcast
             C1>,                                            //   VisitorAuxLoad
           C2>>                                              //   VisitorAuxLoad
```

三个观察：

**1. 所有需要访存的节点共用同一个 `OutputTileThreadMap`。**

```cpp
using OutputTileThreadMap = OutputTileThreadLayout<
    ThreadblockShape, WarpShape, ElementC, AlignmentC, EVTEpilogueStages>;   // :177
```

`Bias` / `C1` / `C2` / `D` 四个节点都吃它。这不是巧合——`02` 篇说过 thread map 决定了
`partition()` 出来的六维 tensor 形状，节点之间要能在同一个 `frg_idx` 上对齐，thread map
必须一致。example 15 里出现了两个 thread map（`BiasTileThreadMap` 和
`OutputTileThreadMap`，`:106`/`:127`），但它们只差 `Element` 和 `Alignment`——因为 bias 是
`float` 而输出是 `half_t`，向量宽度不同。

**2. `StrideMNL` 编码了「这是什么形状的张量」。**

```cpp
Bias : cute::Stride<_0, _1, int32_t>          // M 方向 stride 0 → 沿 M 广播的行向量
C1/C2: cute::Stride<int64_t, _1, int64_t>     // 完整矩阵，第三维是 batch stride
```

`_0{}` 是编译期 0，直接让地址计算里那一项消失。所以「广播」不是运行时判断，是 layout
层面的事——这正是论文 5.2 节 Shape/Type Propagation 那个 pass 干的事
（`(4,64):(64,1)` → `(128,4,64):(0,64,1)`）在手写代码里的样子。

**3. `Arguments` 的嵌套顺序是「树的后序 + node op 最后」。**

```cpp
typename EVTD::Arguments callback_args{                                    // :439
  {                                                       // EVTCompute2
    {                                                     //   EVTCompute1
      {                                                   //     EVTCompute0
        {},                                               //       Accum
        {tensor_Vector.device_data(), ElementC(0), {_0{}, _1{}, int32_t(n)}},  // Bias
        {}                                                //       Compute0  ← node op 在最后
      },
      {tensor_c1.device_data(), ElementC(0), {n, _1{}, mn}},  //     C1
      {}                                                  //     Compute1
    },
    {tensor_c2.device_data(), ElementC(0), {n, _1{}, mn}},   //   C2
    {}                                                    //   Compute2
  },
  {tensor_d.device_data(), {n, _1{}, mn}},                // D
};
```

写树时是 `Sm80EVT<NodeOp, Child0, Child1>`，写 arguments 时是
`{child0_args, child1_args, nodeop_args}`。原因在 `02` 篇：`TreeVisitor2x` 继承
`VisitorImpl2x<ChildOps..., NodeOp>`，参数顺序被翻转过，`Params` 是按存储顺序的 tuple。
这是最容易写错的地方，`{}`（compute 节点无参）放错位置编译能过但结果错。

**4. StreamK 是免费的。** `EVTKernelStreamK` 只是把 swizzle 换成
`ThreadblockSwizzleStreamK`（`:254`），`DefaultGemmWithVisitor::SelectBase` 的 SFINAE 自动
选到 `GemmWithEpilogueVisitorStreamk`。visitor 树完全不知道 StreamK 存在——`02` 篇说的
`EpilogueWithVisitorCallbacks::reduce()` 那条独立入口重放了同一套回调。

### 案例二：example 15 —— 最小可用形态

```text
EVTOutput = Sm80EVT<Output, Sm80EVT<ApplyBias, Accum, Bias>>
```

两层，就是 `D = acc + bias`（bias 是完整矩阵，走 `VisitorAuxLoad`）。值得看的是这一行注释
（`:145`）：

```cpp
// Use element type in EVT with the smallest bitwidth as ElementC.
using ElementC = ElementComputeEpilogue;
```

`DefaultGemmWithVisitor` 那个「借一遍 `LinearCombination` 拿 `GemmBase`」的技巧
（`02` 篇）意味着 `ElementC` / `AlignmentC` 仍然参与 default epilogue 的推导——smem 大小、
`SharedLoadIterator` 都跟它有关，即使 EVT 路径下 `ptr_C` 根本不用。所以 `ElementC` 要按
「树里最窄的元素类型」填，否则 smem 分配和向量宽度会算错。

另一件事：这个例子的 mainloop 是 **2:4 结构化稀疏**（`SparseGemmWithVisitor`），说明 EVT
的 epilogue 与 mainloop 完全解耦——论文 5.1 节「保留专家设计的主循环」在这里是字面意义的，
换个稀疏 mainloop，epilogue 树一行不改。

### 案例三：example 49 —— Sm90 的两种用法

`49_collective_builder.cu` 用一个 `bool UseCustomEVT` 模板参数在两条路线间切换：

```cpp
// :291-297  路线 B：手写树
using CustomEVT =  // alpha * acc + beta * C
  Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementD, ElementCompute, RoundStyle>,
    Sm90ScalarBroadcast<ElementScalar>,                   // beta
    Sm90SrcFetch<ElementC>,                               // C
    Sm90EVT<Sm90Compute<multiplies, ElementCompute, ElementCompute, RoundStyle>,
      Sm90ScalarBroadcast<ElementScalar>,                 // alpha
      Sm90AccFetch>>;                                     // acc

// :315  交给 builder
cute::conditional_t<UseCustomEVT, CustomEVT, DefaultOperation>
```

对比 2x 要记两点差别：

- **`Sm90SrcFetch` 取的是 collective 已经用 TMA 加载好的 C**，不是自己发 load。2x 没有
  这个节点，C 得当 aux tensor 自己 load。
- `beta * C + (alpha * acc)` 用一个 `homogeneous_multiply_add` 的三元 compute 节点表达。
  `homogeneous_*` 前缀在两条路径上都是必需的技巧：`VisitorCompute` / `Sm90Compute` 的
  `ComputeFn` 槽是 `template <class> class`，而 `cutlass::multiply_add` 有三个模板参数，
  匹配失败；`cutlass::homogeneous_multiply_add`（`functional.h`）是单参包装。本仓库
  `evt_attention_scores.h:124-129` 那条注释记的就是这个坑。

### 案例四：example 35/37 —— EVT 之前，以及 EVT 之外

这两个 example 用的是 `epilogue_with_visitor.h` 里那个更老的 `EpilogueWithVisitor`：
epilogue 接受**一个用户手写的 visitor 类**，它填 `begin_epilogue / begin_step / begin_row /
visit / end_row / end_step / end_epilogue` 七个钩子。钩子名字和 EVT 完全一样（论文 Fig. 9
的命名），但没有节点库、没有树。

**example 37（GEMM + LayerNorm）** 的 `EpilogueVisitorLayerNorm`
（`gemm_with_layernorm.h:451`）：`visit` 里累加 `accum_sum_element_` 和
`accum_sum_square_`，`end_row`（`:563`）里

```cpp
bool is_write_thread = (thread_offset_.row() < extent_.row() && (threadIdx.x % kThreadsPerRow) == 0);
arch::global_store<ElementVariance, ...>(convert_variance_output(accum_sum_square_), ...);
arch::global_store<ElementMean, ...>(convert_mean_output(accum_sum_element_), ...);
```

注意它写的是 `ptr_Variance` / `ptr_Mean` 两个**独立输出**，不是归一化后的结果——这是
**partial** 归约，后面还要一个 kernel 做 finalize + 应用。它用 `Σx` 和 `Σx²` 而不是两遍
均值/方差，正是论文表 1 里 BN 分解那个 `σ = sqrt(ΣX²/n - (ΣX/n)²)` 的思路。

**example 35（GEMM + softmax）** 的 `EpilogueVisitorSoftmax`
（`include/cutlass/epilogue/threadblock/epilogue_visitor_with_softmax.h:65`）更值得读，因为
它就是「在 epilogue 里能把 softmax 做到什么程度」的答案。`visit`（`:328-362`）里做的是
**online softmax 的 running max/sum 更新**：

```cpp
ElementSoftmaxCompute accum_max_prev = accum_max_;
... accum_max_ = maximum_accumulator_(result, accum_max_);
accum_max_ = warp_reduce_max_(accum_max_);                        // :345
ElementSoftmaxCompute updater = fast_exp(accum_max_prev - accum_max_);   // :347
SoftmaxFragment intermediate = exponential(minus(result, accum_max_));
accum_sum_ = sum_accumulator_(intermediate, accum_sum_ * updater);       // :359
```

`end_row`（`:373-402`）`warp_reduce_sum_` 之后把 `accum_max_` 和 `accum_sum_` 写去
`ptr_Max` / `ptr_Sum`。然后 device 层（`gemm_with_softmax.h:577-620`）launch **三个 kernel**：

```text
GemmKernel                  → 出 D（未归一化）+ partial max/sum
ApplyFinalReductionKernel   → 跨 CTA 合并 partial max/sum
SoftmaxApplyKernel          → 读 D + 最终 max/sum，出归一化结果
```

这就是 `02` 篇那个结论的实证：epilogue 能算 partial max/sum，但「拿最终归约结果回头改本
tile 每个元素」需要重新读一遍 D，只能是另一个 kernel。CUTLASS 里最会写 epilogue 的人，
在 N 跨多个 CTA 时也只能这么做。

顺带说清一件事：example 35/37 的 visitor **不是 EVT**。它们是 EVT 的前身，也是「当融合模式
不在节点库里、且需要多 kernel 协作时」的正确写法。论文 5.1 节讲的可扩展性问题就是它们暴露
的问题——每个新模式手写一个 visitor 类。

### 案例五：本仓库的 EVT 实例

`csrc/swin/kernel/evt_attention_scores.h`，算 Swin window attention 的 QK^T 阶段：

```text
scores[b] = scale · (Q[b] · K[b]^T) + rel_pos_bias[b]
```

树（`:108-142`）：

```text
EVTStore = Sm80EVT<Store,                                   // VisitorAuxStore -> scores
             Sm80EVT<ComputeScaleBias,                      // homogeneous_multiply_add
               ScaleBroadcast,                              // VisitorScalarBroadcast(scale)
               Accum,                                       // VisitorAccFetch  (Q.K^T)
               BiasAux>>                                    // VisitorAuxLoad   (rel_pos_bias)
```

`multiply_add(a, b, c) = a*b + c`，三个 child 按顺序就是 `(scale, acc, bias)`。

问题映射（`:39-49`）：batched GEMM，一个 batch = 一个 (window, head)：

```text
C[M=L, N=L] = A[M=L, K=head_dim] · B[K=head_dim, N=L]
A = Q[b]   RowMajor,    lda = head_dim
B = K[b]   ColumnMajor  →  等价于 K^T，ldb = head_dim
D = scores[b]           RowMajor, ldd = L_pad
```

`LayoutB = ColumnMajor` 是这里最漂亮的一手：K 在内存里是 `[L, head_dim]` row-major，声明成
column-major 读就直接得到 `K^T`，**不需要任何 transpose kernel**。这正是论文 5.3 节
"Mainloop Fusion" 说的那件事——「用 stride 表达 permutation，不真的搬数据」。

这个实例踩过的四个坑（都记在文件注释和 memory 里，值得复述因为都是 EVT 特有的）：

1. **`ComputeFn` 槽的模板参数个数**：用 `cutlass::homogeneous_multiply_add`，不是
   `cutlass::multiply_add`（后者 3 个模板参数，匹配 `template <class> class` 失败）。
2. **`OutputTileThreadLayout` 的头文件位置**：在
   `epilogue/threadblock/fusion/visitor_2x.hpp`（经 `visitors.hpp` 引入），不在名字更像的
   `output_tile_thread_map.h`。
3. **`GemmUniversalAdapter::Arguments` 没有 `problem_shape` 字段**：kernel 内部自己推
   `{M, N, batch_count}`（`gemm_universal_with_visitor.h:139`），调用方只要传对
   `batch_count`，aux load/store 靠 `StrideMNL` 的 mode-2 分量做 batch 偏移。
4. **`kAlignmentC` 必须是 1**：对齐检查针对**逻辑 N = L = 49**（不是 8 的倍数），而不是
   storage 的 `L_pad = 56`。A/B 沿 `head_dim = 32` 可以用 8。否则报
   `Error Misaligned Operand`。

第 4 条值得多说一句：`L_pad` 的存在是为了让 8 宽的 epilogue 向量写不越界，但 GEMM 的 N
extent 仍然是 49，`VisitorAuxStore` 的 predicate 用 `elem_less(coord, problem_shape)` 判断，
所以 `[L, L_pad)` 那段尾列不会被写——host reference 也要把它留成 0 才能对齐
（`csrc/tests/swin/evt_attention_scores.cu:168-171`）。

验证状态：`csrc/tests/swin/evt_attention_scores.cu`，target `swin_evt_attention_scores`
（`csrc/swin/CMakeLists.txt:84`），host reference 算 `scale·QK^T + bias`（不含 softmax），
**MAE 7.66e-9，RTX 4070 / SM89 通过**。

### 从案例回看抽象

把五个案例横过来看，EVT 的实际适用边界很清楚：

| 模式 | EVT 能否单核融合 | 实例 |
| --- | --- | --- |
| `acc` 的任意 elementwise 组合 | ✅ | ex47、ex15、ex49 |
| 标量 / 行 / 列广播 | ✅ | ex47 的 Bias（行广播） |
| 完整 aux tensor 逐元素参与 | ✅ | ex47 的 C1/C2、本仓库的 bias |
| 多输出（同时写 D 和中间量） | ✅ store 是透传节点 | 论文 Fig. 11 |
| 沿 N 归约成列向量 | ⚠️ partial + atomic | `VisitorColReduction` |
| 沿 M 归约成行向量 | ⚠️ partial + smem + atomic | `VisitorRowReduction` |
| 归约结果回头修改本 tile | ❌ 2x；✅ 3x 且归约维在单 warp 内 | `Sm90TopKSoftmax` |
| 完整 row-softmax（N 跨多 CTA） | ❌ 必须多 kernel | ex35 三 kernel |
| epilogue 里再发一次 matmul | ❌ | 要 B2B（ex13）或 monolithic（ex41） |

最后一行值得点明：**example 41（fused multi-head attention）的 bias + softmax 不是 EVT**，
它是 monolithic kernel 里的自定义 fragment 代码。同理 `csrc/flash-attention` 那条链路也不是
EVT。把 attention 做成单 kernel 和用 EVT 融合 epilogue 是两种不同的技术，`05` 篇要在这个
区分上做设计。

## 遗留问题

- example 47 用了 `EVTEpilogueStages`，本仓库实例用 1；两级 ping-pong 在这种 aux-load 较重
  的树上能带来多少收益，没测过（要测得在 `csrc/swin` 那条链路上做 A/B）。
- `VisitorRowBroadcast` 的 `EnableNullptr` 分支（`visitor_load.hpp:401-410`）允许传 nullptr
  退化成标量填充，我没验证它在树里与 predicate 的交互。
- example 15 那句「ElementC 取最窄类型」我理解为影响 smem 与向量宽度推导，但没做实验确认
  取宽了具体会错在哪。

## Profile

<!-- 本工作区不新增 kernel。已验证实例的 NCU 数据归属 csrc/swin，待其 bench 通过后补。 -->

## 后记

<!-- HUMAN: 读完补充自己的结论、踩坑和下一步。 -->
