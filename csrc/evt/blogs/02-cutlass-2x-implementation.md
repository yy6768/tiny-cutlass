# EVT 学习记（二）：CUTLASS 2x 路径的实现原理

## 前言

`00` 篇里把 EVT 分成了「论文的编译器系统」和「CUTLASS 里的 visitor 模板」两层。这一篇读
第二层在 Ampere/Ada（CUTLASS 2x epilogue）上的实现：一棵 `Sm80EVT` 到底展开成了什么、
`visit` 的参数为什么是 `(iter_idx, row_idx, column_idx, frg_idx, frg_acc)`、四类节点各自
用到了哪些回调钩子、以及归约节点是怎么绕过「跨 CTA 归约」这件事的。

读的源码（都在 `3rdparty/cutlass/include/cutlass/`）：

- `epilogue/threadblock/fusion/visitor_2x.hpp` — tree / topological visitor 与线程布局
- `epilogue/threadblock/fusion/visitor_load.hpp` — 五个 load 节点
- `epilogue/threadblock/fusion/visitor_compute.hpp` — 唯一的 compute 节点
- `epilogue/threadblock/fusion/visitor_store.hpp` — store 与三个归约节点
- `epilogue/threadblock/epilogue_with_visitor_callbacks.h` — 驱动这棵树的 epilogue
- `gemm/kernel/gemm_universal_with_visitor.h` / `default_gemm_universal_with_visitor.h`

## Overview

### 装配链

从外到内是这么一条链（行号为当前 checkout）：

```text
DefaultGemmWithVisitor<..., FusionCallbacks, ThreadblockSwizzle, Stages, Operator, EpilogueStages>
                                              default_gemm_universal_with_visitor.h:102
  ├─ GemmBase = DefaultGemmUniversal<..., LinearCombination<...>, ...>::GemmKernel   :104
  │    ↑ 借用普通 GEMM 的装配结果拿 Mma 和 "DefaultEpilogue 描述子"
  ├─ Epilogue = EpilogueWithVisitorCallbacks<GemmBase::Epilogue, FusionCallbacks, EpilogueStages>
  │                                             default_gemm_universal_with_visitor.h:123
  └─ GemmKernel = SelectBase<ThreadblockSwizzle>                                     :148
       ├─ 普通 swizzle  -> GemmWithEpilogueVisitor<Mma, Epilogue, Swizzle>
       └─ StreamK swizzle -> GemmWithEpilogueVisitorStreamk<...>   （SFINAE on StreamkFeature）
```

有意思的是 `GemmBase`：它先用一个**丢弃不用的** `LinearCombination` 走一遍
`DefaultGemmUniversal`，只为了拿到 `Mma` 和一个「默认 epilogue 描述子」——后者的
`Shape` / `WarpMmaOperator` / `AccumulatorFragmentIterator` / `WarpTileIterator` /
`SharedLoadIterator` / `OutputTileIterator` 被 `EpilogueWithVisitorCallbacks` 当作
`DefaultEpilogue` 参数逐个取用（`epilogue_with_visitor_callbacks.h:66-115`）。也就是说
EVT **没有重写 accumulator→smem→寄存器这套搬运机制**，它只替换了「拿到寄存器里的输出
fragment 之后干什么」。这是这条路径最省事、也最限制它的地方。

kernel 侧唯一的额外状态是 `problem_shape`：

```cpp
// gemm_universal_with_visitor.h:107
cute::Shape<int32_t,int32_t,int32_t> problem_shape;
// :139
problem_shape({args.problem_size.m(), args.problem_size.n(), args.batch_count}),
// :311
epilogue(accumulators, threadblock_tile_offset, params.problem_shape, thread_idx);
```

`(M, N, L)` 三元组，L 就是 batch_count。所有 visitor 节点的 `StrideMNL` 都是相对这个
shape 的 stride，这解释了为什么 aux load/store 的 batch 偏移是 mode-2 的 stride，而调用侧
不需要自己传 `problem_shape`（本仓库 `evt_attention_scores.h:228` 那条注释踩的就是这个坑）。

### 树是怎么展开的

`Sm80EVT` 只是别名（`visitor_2x.hpp:317`）：

```cpp
template <class NodeOp, class... ChildOps>
using Sm80EVT = TreeVisitor2x<NodeOp, ChildOps...>;
```

`TreeVisitor2x` 继承 `VisitorImpl2x<ChildOps..., NodeOp>` —— 注意**参数顺序翻转了**：
写代码时 node op 在前、children 在后，存储时 children 在前、node op 在最后一个 slot。
`VisitorImpl2x` 本身继承 `fusion::detail::Sm90VisitorImplBase<Ops...>`
（`visitor_2x.hpp:53`），也就是 2x 复用了 3x 的「一个 tuple 装所有 op」的基础设施。

`visit` 的递归就是一个 `tapply`（`visitor_2x.hpp:185-199`）：

```cpp
constexpr int Rm1 = sizeof...(ChildOps);
return cute::detail::tapply(callbacks_tuple,
  [&] (auto& child_callbacks) {                       // transform: 先算每个 child
    return child_callbacks.visit(iter_idx, row_idx, column_idx, frg_idx, frg_acc);
  },
  [&] (auto&&... frg_inputs) {                        // apply: 结果作为额外参数喂 node op
    return get<Rm1>(callbacks_tuple).visit(iter_idx, row_idx, column_idx, frg_idx,
                                          frg_acc, frg_inputs...);
  },
  make_seq<Rm1>{}
);
```

三件事值得记住：

1. **child 必须是 nullary 的**——它的 `visit` 只接受 `frg_acc`，不接受额外输入。所以
   child 只能是 load 节点或另一棵完整子树；`VisitorCompute` 只能出现在 node op 位置。
2. **`frg_acc` 被广播给整棵树**。每个节点都能看到原始 accumulator，这就是 `VisitorAccFetch`
   （`visitor_load.hpp:59`）能实现成「直接 `return frg_acc`」的原因，也是为什么树里可以
   多处引用 accumulator 而不需要额外存储。
3. **递归发生在编译期，运行时是完全内联的表达式**。没有虚函数、没有间接跳转，「visitor」
   在这里是编译期模式而不是运行时多态。

出度 > 1 的节点用 `TopologicalVisitor2x`（`visitor_2x.hpp:233`）：它的模板参数多一个
`EdgeTuple`，`visit` 里先为前 R-1 个 op 各准备一个 `Array<ElementCompute, FragmentSize>`
寄存器 buffer，按拓扑序算，每个 op 通过 `edge_seq` 从 buffer tuple 里挑自己的输入
（`visitor_2x.hpp:253-286`）。这正是论文 Fig. 8(E) 描述的 topological visitor：
「用临时寄存器 buffer 保存各输入节点的中间结果」。

> 一个可以顺手确认的点：`TopologicalVisitor2x` 要求 `EdgeTuple` 是 static 的、rank 等于
> op 数、op 数 > 1（`visitor_2x.hpp:234-236`），也就是边集在编译期完全已知。这与论文里
> 「组合爆炸发生在边集上」的说法是一致的——边集是模板参数，节点实现是库。

### 回调协议：一次 epilogue 的时间轴

`EpilogueWithVisitorCallbacks::operator()` 有两条分支（`Pipelined = Stages > 1`），
非 pipelined 分支（`epilogue_with_visitor_callbacks.h:391-486`）最容易读：

```text
callbacks = fusion_callbacks.get_callbacks(threadblock_tile_offset, thread_idx, problem_shape)
begin_epilogue()
for iter_idx in [0, kIterations):
    begin_step(iter_idx)
    __syncthreads()
    acc2smem_source_needed::push(iter_idx, accum_fragment_iterator, warp_tile_iterator_)
    __syncthreads()
    shared_load_iterator_.load(aligned_accum_fragment[0])       // + kPartitionsK 归约
    for idx in [0, kAccumulatorFragmentCount):
        row_idx = idx / Iterations::kColumn
        col_idx = idx % Iterations::kColumn
        if col_idx == 0: begin_row(row_idx)
        visit(iter_idx, row_idx, col_idx, idx, accum_frag_ptr[idx])
        if col_idx + 1 == Iterations::kColumn: end_row(row_idx)
    end_step(iter_idx)
end_epilogue()
```

对照论文 Fig. 9 的 `begin_epilogue / begin_step / begin_row / visit / end_row /
end_step / end_epilogue`——一一对应，名字都没改。论文里那句「所有算子都可以只填 `visit`
实现，但利用其他代码块能获得更好性能」在这里是字面意义的：`VisitorImpl2x::Callbacks` 给
六个钩子提供了「转发给所有 child」的默认实现，只有 `visit` 是 `= delete`
（`visitor_2x.hpp:98`），必须每个节点自己实现。

`Stages == 2` 的 pipelined 分支（`:286-389`）多做的事是论文 5.3 节说的 ping-pong buffer：
`SharedStorage` 里放 `Base::SharedStorage acc_smem[Stages]`（`:147`），
`warp_iterator_offset` / `smem_iterator_offset` 每轮取反在两块 smem 之间来回切
（`:299-300`、`:345-346`），并且**把 `begin_step(iter_idx)` 提前到本轮 smem 写入之前**，
让 aux load 的 global 访存与上一轮的 smem 读+计算重叠。注意 `Stages <= 2` 是硬断言
（`:84`，`visitor_2x.hpp:356` 也有一份），本仓库的 EVT 实例用的是 `kEpilogueStages = 1`。

另外 `reduce()`（`:178-256`）是 StreamK 的 fixup 路径：先用 `BaseStreamK::reduce` 把 peer
block 的 partial accumulator 合起来，再走同一套 `begin_epilogue → begin_step → visit →
end_step → end_epilogue`。所以 EVT 与 StreamK 兼容（论文 5.3 节）不是靠 visitor 节点适配，
而是靠这条独立入口重放同一个回调序列。

### 四类节点各自用了哪些钩子

这是把论文 Fig. 9 落到代码的表。`visit` 每个节点都有，下表只列它**额外**用到的钩子：

| 节点 | 源码 | 额外钩子 | 干什么 |
| --- | --- | --- | --- |
| `VisitorAccFetch` | `visitor_load.hpp:59` | — | `return frg_acc` |
| `VisitorScalarBroadcast` | `:94` | —（在 `get_callbacks` 里取标量） | `frg_scalar.fill(scalar)` |
| `VisitorRowBroadcast` | `:341` | `begin_epilogue` | 整个 epilogue 只需读一次行向量 |
| `VisitorColBroadcast` | `:481` | `begin_epilogue` | 同上，列向量 |
| `VisitorAuxLoad` | `:209` | `begin_step` | 每 step 预取一块 aux tile 到寄存器 |
| `VisitorCompute` | `visitor_compute.hpp:62` | — | 纯计算 |
| `VisitorAuxStore` | `visitor_store.hpp:62` | `begin_step` / `end_step` | 攒寄存器，step 末 `global_store` |
| `VisitorColReduction` | `:251` | `begin_row` / `end_row` | 行内归约 + warp shuffle + atomic |
| `VisitorRowReduction` | `:397` | `begin_epilogue` / `end_epilogue` | 列向累加 + smem 转置归约 + atomic |
| `VisitorScalarReduction` | `:670` | — | 全 tile 归约 |

几个细节值得单独说。

**`VisitorAuxLoad` 的 `begin_step` 就是论文 Fig. 9(A)。** 代码
（`visitor_load.hpp:272-283`）：

```cpp
begin_step(int step_idx) {
  clear(tC_rAux(_,_,_,step_idx%Stages));
  auto src_v   = filter(tC_gAux(_,_,_,step_idx));
  auto coord_v = filter(tC_cAux(_,_,_,step_idx));
  auto dst_v   = filter(tC_rAux(_,_,_,step_idx%Stages));
  for (int i = 0; i < size(src_v); ++i) {
    bool guard = elem_less(coord_v(i), problem_shape);
    cutlass::arch::global_load<VecType, sizeof(VecType)>(dst_v(i), &src_v(i), guard);
  }
}
```

`visit` 只是从寄存器 tile 里取对应 fragment（`:285-291`）。三点：向量宽度由
`ThreadMap::kElementsPerAccess * sizeof_bits<Element>` 上限 128 bit 决定（`:237-239`）；
越界用 `make_identity_tensor` 生成坐标张量 + `elem_less(coord, problem_shape)` 做 predicate，
不是靠 clamp 地址；`step_idx % Stages` 就是 ping-pong。

**`VisitorAuxStore` 是「透传节点」。** 论文说 store 被建模成转发输入的透传节点，代码里
就是字面这样（`visitor_store.hpp:126-136`）：`visit` 把转换后的值写进寄存器 tile 然后
`return frg_input`（返回的是**输入**，不是转换后的值），真正的 `global_store` 在
`end_step`。所以一棵树里可以有多个 store 节点串在路径上，各自落盘而不打断数据流——EVT
支持多输出（论文 Fig. 11 的 `reduce` 与 `out` 两个输出）靠的就是这个性质。

**两个向量归约走的是完全不同的策略**，这是这一篇最值得记的对比：

`VisitorColReduction`（归约成列向量，即沿 N 归约、每行一个值）——归约方向和线程布局同向，
所以只需要 warp 内 shuffle：

```cpp
begin_row(row_idx)  : reduction_accum = reduction_identity            // :313-316
visit(...)          : if (coord_n < n) fragment_reduce(reduction_accum, frg_input);
                      if (column_idx + 1 == Iterations::kColumn)
                        intra_warp_row_reduce<RegReduceFn, kAccessWidth>(reduction_accum);
                                                                       // :318-347
end_row(row_idx)    : if (guard && is_writing_thread)
                        atomic_reduce(&tC_gCol(row_idx, curr_iter_idx), reduction_accum);
                                                                       // :349-356
```

`is_writing_thread = thread_idx % kAccessWidth == 0`（`:301-303`），即每行只有一个线程发
atomic。

`VisitorRowReduction`（归约成行向量，沿 M 归约）——归约方向与线程布局垂直，必须过 smem：
节点自带 `SharedStorage`（`:420-423`），`visit` 只往寄存器 buffer 里按列累加
（`:485-500`），`end_epilogue` 里 `__syncthreads` → `copy(tRS_rSrc, tRS_sRows)` →
`__syncthreads` → 换一套「每线程负责若干列、扫所有行」的映射做第二轮归约 → `atomic_reduce`
（`:504-547`）。

两者的共同点是**最后都是 atomic**。这正是 EVT 归约能力的实质：epilogue 只能把**一个 CTA
tile 内**的部分归约做完，跨 CTA 的合并交给 global atomic（所以调用方必须先把输出清零，
`reduction_identity` 也只是 CTA 内的初值）。论文 3.1 节「归约需要所有待归约元素被划分到
同一 threadblock，这与 GEMM 主循环 tiling 不兼容」，代码给出的解法就是「不强求，用
atomic」。

**这也直接给出了 softmax 融不进去的机制性原因**：softmax 需要
`max → exp → sum → divide`，即在得到完整行归约结果**之后**还要回头修改这一行的每个元素。
atomic 让归约结果只在所有 CTA 都写完后才正确，而写 atomic 的那个 CTA 此时早已把
`frg_acc` 丢掉了。`VisitorColReduction` 能算出 partial max/sum，但不存在「等所有 partial
到齐再回到本 tile」的钩子——`end_epilogue` 之后 kernel 就结束了。这与 example 35 把
GEMM+softmax 做成三个 kernel 是同一件事的两面。

### `OutputTileThreadLayout`：2x thread map 到 cute layout 的桥

所有需要访问 global 的节点都吃一个 `ThreadMap` 模板参数，实参统一是
`OutputTileThreadLayout<ThreadblockShape, WarpShape, Element, ElementsPerAccess, Stages>`
（`visitor_2x.hpp:340`）。它继承 2x 时代的 `DefaultThreadMapTensorOp<...>::Type`，然后把
后者的标量常量翻译成 cute layout：

```cpp
using ThreadMapShape = cute::Shape<
  Int<kElementsPerAccess>, Int<kAccessWidth>, Int<Iterations::kColumn>,   // 列方向
  Int<kAccessRows>, Int<Iterations::kRow>, Int<kWarpsRemainingForRows>,
  Int<Count::kRow>, Int<Count::kGroup>, Int<Shape::kGroup>,
  Int<Count::kCluster>, Int<Shape::kCluster>>;                            // 行方向
                                                            // visitor_2x.hpp:370-384
```

`partition(xT, thread_idx, threadblock_tile_offset)`（`:410-425`）做三步：`local_tile` 按
`CtaShapeMNL` 取本 CTA 的 tile → `tid2coord` 把 `thread_idx` 拆成
`(lane_col, lane_row, warp_row, group, cluster)` 五维坐标 → 把 tile 转成 column-major 再
`compose(ThreadMapShape)`，最后按线程坐标切片，得到
`(VECTOR, FRAGMENT_COLUMN, FRAGMENT_ROW, ITERATION_ROW, ITERATION_GROUP, ITERATION_CLUSTER)`。

这就是「2x epilogue 的 thread map 用 3x 的 cute 语言重新表达」——节点实现只跟 cute tensor
打交道，不用知道 2x 那堆 `Iterations::kColumn` 常量。它也解释了本仓库踩到的那个坑：
`OutputTileThreadLayout` 在 `visitor_2x.hpp` 里，不在名字更像的
`output_iterator_thread_map.h`；以及 `Stages <= 2` 的断言在这里也有一份（`:356`）。

### 与 visitor 前身的关系

`epilogue_with_visitor.h` 里的 `EpilogueWithVisitor`（`:167`）是更早的一层：它同样调用
`visitor.begin_epilogue() / begin_step / begin_row / visit / end_row / end_step /
end_epilogue`（`:253-349`），但 visitor 是**用户手写的单个类**，不是可组合的树。
example 35 的 `EpilogueVisitorSoftmax` 和 example 37 的 `EpilogueVisitorLayerNorm`
（`gemm_with_layernorm.h:451-599`）就是这种手写 visitor。

所以 CUTLASS 里 EVT 的演进是两步：先有「epilogue 可以接受一个填钩子的 visitor 类」
（example 35/37），再有「visitor 由节点库 + tree 在编译期组装」（`fusion/` 目录）。论文
5.1 节讲的可扩展性问题，正是第一步留下的问题——每个新融合模式要手写一个 visitor 类。

## 遗留问题

- `VisitorScalarReduction`（`visitor_store.hpp:670`）我只确认了它存在和签名，没读它的
  归约路径细节；它是否也走 smem，待读。
- `EpilogueWithVisitorCallbacks` 的 `kAccumulatorFragmentCount`（`:144`）与
  `SharedLoadIterator::ThreadMap::Iterations::kColumn` 的关系我按「row-major 展开」理解，
  没有实验验证过 `row_idx` / `col_idx` 与实际输出坐标的对应；要确认得写一个只打印坐标的
  visitor 节点。
- conv 侧：`include/cutlass/conv/` 下 grep 不到 `EpilogueWithVisitor` 或 `FusionCallbacks`，
  即 2x 的 implicit GEMM conv kernel 不走 EVT 路径。这与 `csrc/swin/docs/00-overview.md`
  里「conv fprop 本身不支持 epilogue-visitor，要 fork 一个 conv kernel」的判断一致，但我
  没有逐个 conv kernel 确认是否存在别的入口。

## Profile

<!-- 本工作区不新增 kernel，无实测数据。 -->

## 后记

<!-- HUMAN: 读完补充自己的结论、踩坑和下一步。 -->
