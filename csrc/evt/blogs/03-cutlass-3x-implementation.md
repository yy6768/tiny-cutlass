# EVT 学习记（三）：CUTLASS 3x（Sm90+）路径的实现原理

## 前言

`02` 篇读的是 Ampere/Ada 上的 2x epilogue。这一篇读 Hopper 之后那套：为什么同一个抽象要
写第二遍、`Sm90EVT` 与 `Sm80EVT` 差在哪、多出来的 `previsit` / `reduce` / `postreduce`
三个钩子是为了什么，以及 `Sm90TopKSoftmaxColReduction` 这个「看起来能做 softmax」的节点
到底做了什么、为什么它不是 2x 缺失能力的补齐。

读的源码（`3rdparty/cutlass/include/cutlass/epilogue/`）：

- `fusion/sm90_visitor_tma_warpspecialized.hpp` — impl base、tree/split-tree/topological visitor
- `fusion/sm90_visitor_load_tma_warpspecialized.hpp` — load 节点
- `fusion/sm90_visitor_store_tma_warpspecialized.hpp` — store 与归约节点
- `fusion/sm90_visitor_compute_tma_warpspecialized.hpp` — compute 节点
- `fusion/sm90_visitor_topk_softmax.hpp` — Top-K + softmax 归约
- `fusion/callbacks.hpp` / `fusion/operations.hpp` — `FusionCallbacks` 派发层与预置融合操作
- `collective/sm90_epilogue_tma_warpspecialized.hpp` — 驱动这棵树的 collective epilogue

## Overview

### 为什么要写第二遍

2x 的 epilogue 数据流是固定的：accumulator → smem 换 layout → 寄存器 → 逐 fragment 调
`visit` → global store。Hopper 的 epilogue 不是这个形状：

- 输出侧走 **TMA**，store 要经过 smem 并且是异步的、带 pipeline 的；
- source（C 矩阵）和 aux tensor 的加载由**独立的 producer warp** 用 TMA 预取，与 consumer
  warp 的计算 warp-specialize 开来；
- 因此「读输入」和「写输出」是两条不同的回调链，节点必须同时提供两套。

所以 3x 的节点接口是双份的（`sm90_visitor_tma_warpspecialized.hpp`）：

```text
is_producer_load_needed() / is_C_load_needed()        编译期+运行期查询：要不要 producer 参与
get_producer_load_callbacks(ProducerLoadArgs)   -> ProducerLoadCallbacks   { begin, step, end }
get_consumer_store_callbacks<ReferenceSrc>(ConsumerStoreArgs)
                                                -> ConsumerStoreCallbacks { begin, begin_loop,
                                                     previsit, visit, reduce, postreduce,
                                                     end_loop, end }
```

`ProducerLoadArgs`（`:290`）和 `ConsumerStoreArgs`（`:328`）这两个参数包本身就说明了差别：
2x 只传 `(threadblock_tile_offset, thread_idx, problem_shape)` 三个标量，3x 传的是
`problem_shape_mnkl` / `tile_shape_mnk` / `tile_coord_mnkl` / `tiled_mma` / `epi_tile` /
`tiled_copy` / 坐标张量 `cD` / residue / 线程切片后的坐标 `tCcD` / `tCrC` 引用。节点能拿到
完整的 cute 布局信息，代价是它必须自己做 partition。

### 树的结构：与 2x 同形

`Sm90EVT` 也只是别名（`sm90_callbacks_tma_warpspecialized.hpp:58`）：

```cpp
template <class NodeOp, class... ChildOps>
using Sm90EVT = Sm90TreeVisitor<NodeOp, ChildOps...>;
```

`Sm90TreeVisitor`（`sm90_visitor_tma_warpspecialized.hpp:573`）继承
`Sm90VisitorImpl<ChildOps..., NodeOp>`，`visit` 的递归和 2x 一模一样，只是坐标参数换了：

```cpp
// :595-608
visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n) {
  constexpr int Rm1 = sizeof...(ChildOps);
  return cute::detail::tapply(callbacks_tuple,
    [&] (auto& child_callbacks) { return child_callbacks.visit(frg_acc, epi_v, epi_m, epi_n); },
    [&] (auto&&... frg_inputs) {
      return get<Rm1>(callbacks_tuple).visit(frg_acc, epi_v, epi_m, epi_n, frg_inputs...);
    },
    make_seq<Rm1>{});
}
```

`(iter_idx, row_idx, column_idx, frg_idx)` 变成了 `(epi_v, epi_m, epi_n)`——epilogue subtile
的两维索引加 fragment 内的向量索引。`02` 篇那个「一个线程只看得到自己那几个元素 + 自己在
tile 里的位置」的结论在这里完全不变，只是坐标系换了。

2x 里 `VisitorImpl2x` 直接继承 `fusion::detail::Sm90VisitorImplBase`
（`visitor_2x.hpp:53`），也就是这套「一个 tuple 装所有 op、`for_each` 转发钩子」的骨架是
3x 先写的，2x 复用。`Sm90VisitorImplBase` 还给 1/2/3/4 个 op 的情形写了显式特化
（`:755`、`:826`、`:914`、`:1020`），纯粹是为了减少模板实例化开销。

三种 visitor：

| visitor | 源码 | 适用 |
| --- | --- | --- |
| `Sm90TreeVisitor` | `:573` | 出度 1 的普通树 |
| `Sm90SplitTreeVisitor` | `:633` | 共享一个 input tree、多个 output tree 的 DAG |
| `Sm90TopologicalVisitor` | `:681` | 一般 DAG，带 `EdgeTuple` |

`Sm90SplitTreeVisitor` 是 2x 没有的：它先算 input tree，然后**把结果当作 acc fragment
喂给各个 output tree**（`:648-657`）。论文里没有单独讲这个形态，它是「多输出 DAG 的常见
特例」的一个工程优化——比 topological visitor 省寄存器 buffer。

### 回调时间轴：consumer store loop

`sm90_epilogue_tma_warpspecialized.hpp` 的 store 循环（`:778-919`）：

```text
cst_callbacks.begin()
if (cst_callbacks.begin_sync_needed()) __syncthreads()       // 允许跨节点串异步拷贝
for epi_n in [0, size<3>(gD_epi)):
  for epi_m in [0, size<2>(gD_epi)):
     cst_callbacks.begin_loop(epi_m, epi_n)
     ... 等 producer 的 TMA load ...
     cst_callbacks.previsit(epi_m, epi_n, load_wait_state.count(), is_producer_load_needed)
     for epi_v:
        tRS_rCompute_frg(epi_v) = cst_callbacks.visit(tRS_rAcc_frg_mn(...), epi_v, epi_m, epi_n)
     cst_callbacks.reduce(sD_epi(...), sync_fn, epi_m, epi_n, is_last_iteration, tRS_rCompute_frg)
     ... 寄存器 -> smem -> TMA store ...
     cst_callbacks.postreduce(epi_m, epi_n, store_pipe_producer_state.count(), issue_smem_store)
     cst_callbacks.end_loop(epi_m, epi_n)
cst_callbacks.end()
```

对照 2x 的 `begin_epilogue / begin_step / begin_row / visit / end_row / end_step /
end_epilogue`，映射关系大致是：

```text
2x begin_epilogue  ≈ 3x begin
2x begin_step       ≈ 3x begin_loop + previsit
2x visit            ≈ 3x visit
2x begin_row/end_row  没有对应物（3x 没有"行"的概念，只有 epi_m/epi_n subtile）
2x end_step         ≈ 3x postreduce / end_loop
2x end_epilogue     ≈ 3x end
新增：reduce(smem_buffer, sync_fn, ..., visit_results)
```

**`reduce` 是 3x 独有的、也是最关键的新钩子**，它多了两样 2x 没有的东西：

1. `smem_buffer` —— epilogue 的 smem 暂存区（正常用来做 R2S 的那块）在这里被借给归约节点
   当 scratch；
2. `visit_results` —— **整个 epilogue subtile 的 `visit` 结果**，而且是个可写的寄存器
   tensor。

第 2 点是能力上的实质区别。2x 的 `visit` 返回值立刻被父节点消费，节点无法回头改已经算出
的结果；3x 的 `reduce` 拿到的是「本 subtile 全部 visit 结果」的引用，可以先归约再回写。
这就是「先归约、再用归约结果修改本 tile 元素」这类模式在 3x 上成立、在 2x 上不成立的原因。

`sync_fn` 是 epilogue 传进来的同步回调，节点自己决定何时同步；这让归约节点能在
`reduce` 内做 smem 往返而不必知道外层 pipeline 的状态。

### 归约节点：3x 多了非 atomic 的两阶段路径

2x 的三个归约节点最终都是 `atomic_reduce`（见 `02` 篇）。3x 的 `Sm90RowReduction`
（`sm90_visitor_store_tma_warpspecialized.hpp:665`）和 `Sm90ColReduction`（`:1253`）多了
一个模板开关：

```cpp
static constexpr bool IsAtomic = is_atomic<GmemReduceFn<ElementCompute>>::value;
static_assert(not (IsAtomic && not FinalReduction), "atomic reduction must be final");
```

`Params` 里除了 `ptr_row` 还有 `reduction_buffer` 和 `tile_counters`（`:678-684`）。非 atomic
路径的做法（`end()`，`:1112` 起）是：每个 CTA 把自己的 partial 写进 workspace 的
`reduction_buffer`，然后

```cpp
if (thread_idx == 0) *prev_tile_count = atomicAdd(&params.tile_counters[n], 1);
sync_fn();
do_final_reduction = *prev_tile_count == size<2>(gBuf_ml) * size<3>(gBuf_ml) - 1;
```

**最后一个到达的 CTA 负责做 final reduction**——经典的 last-block-does-the-final-pass 模式。
这比 atomic 好在确定性（浮点归约顺序固定）和支持非交换/非 atomic 可表达的归约算子，代价是
要 workspace 和一个 counter。

但要注意：这仍然**没有解决 softmax 的问题**。final reduction 由最后一个 CTA 做，那时其他
CTA 的 `visit` 结果早已写出去了。「归约结果能回到所有元素」这件事，只有在归约维完整落在
**一个** CTA tile 内时才成立。

### `Sm90TopKSoftmaxColReduction`：能做的那种 softmax

`sm90_visitor_topk_softmax.hpp:335`。它是 CUTLASS 里唯一一个名字里带 softmax 的 EVT 节点，
所以值得说清它的适用条件——这也是 `05` 篇设计 Swin attention 时最需要的一条事实。

它的工作方式：

```cpp
// visit (:541-563)：只累积 top-K，不改数据
for (i in FragmentSize)
  if (in bounds) add_element_to_desc_sorted_array(tCrTopK(...).top_k_, frg_I[i]);
return frg_input;                       // 透传

// reduce (:567-...)：跨 lane 归约 top-K，然后重访 visit_results 原地改写
//   butterfly 或 shuffle_up 归约 tCrTopK
//   reduce_final(): 得到 (min of top-K, logsumexp)
//   for epi_v in visit_results:
//     visit_frag[i] = masked_softmax(visit_frag[i], tCrSoftmax(...).min_,
//                                    tCrSoftmax(...).logsumexp_)
```

`reduce` 里那句「re-visit and apply top-K and softmax」直接原地改写了 `visit_results`。
这确实是**完整的**（不是 partial 的）softmax。它凭什么能做到？源码注释自己写明了前提
（`:716-718`）：

> We're assuming all elements in a row (over which we're performing the reduction) are
> visited in the same corresponding epilogue tile, and this is what allows us to apply the
> top-K + softmax operation within `reduce()`, by re-visiting the accumulated results.

以及一条硬 static_assert（`:712-713`）：

```cpp
// Make sure there's only one warp across N so we can use warp shuffle intrinsics for reduction.
static_assert(decltype(size<1>(warp_layout_MN))::value <= 1);
```

也就是说：**整个归约维必须落在同一个 epilogue subtile、同一个 warp 内**。加上
`TopK == 2 || TopK == 4` 的限制（`:337-339`，注释说其他 K 值可以放开但有严重性能影响）和
FP32 累加要求，这个节点是为「MoE router 那种 N 很小的 top-K + softmax」定制的，不是通用的
row-softmax。

结论钉住：**EVT 能做完整 softmax 的唯一情形是「归约维完整落在单个 CTA tile / 单个 warp 的
N 范围内」**。这不是 2x/3x 的架构差异，是 epilogue tiling 的本质约束；3x 只是因为有了
`reduce` 钩子和 `visit_results`，能在满足这个前提时把它写出来。

### `FusionCallbacks`：预置操作与自定义树的分界

3x 多了一层 2x 没有的派发层（`fusion/callbacks.hpp:56`）：

```cpp
template <class DispatchPolicy, class Operation, class CtaTile_MNK,
          class EpilogueTile_MN, class... Args>
struct FusionCallbacks {
  static_assert(dependent_false<...>, "Could not find a callbacks specialization.");
};
```

`Operation` 是 `fusion/operations.hpp` 里那一长串预置融合操作的标签：`LinearCombination`
（`:116`）、`LinCombEltAct`（`:131`）、`LinCombPerRowBias`（`:161`）、
`LinCombTopKSoftmaxCol`（`:146`）、`ScaledLinCombPerRowBiasEltActAmaxAux`（`:406`）……
每个标签在 `sm90_callbacks_tma_warpspecialized.hpp` 里有一个把它展开成具体 `Sm90EVT` 树的
特化，例如 `Sm90LinCombTopKSoftmaxCol`（`:2622`）。

这意味着 3x 有两种用法：

```text
路线 A（推荐）：CollectiveBuilder + Operation 标签
   → builder 选好 dispatch policy / epilogue tile / copy atom，
     FusionCallbacks 特化把标签展开成 Sm90EVT 树
路线 B（自定义）：直接把手写的 Sm90EVT 传给 collective
   → callbacks.hpp 的 FusionCallbacksTraits<T> 通用版本兜底（:60），
     DispatchPolicy = void、Operation = FusionOperation，
     即"不知道这是什么融合，按裸 visitor 用"
```

example 49 走的是路线 B（`UseCustomEVT`，`49_collective_builder.cu:291-297`）：

```cpp
using CustomEVT =  // alpha * acc + beta * C
  Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementD, ElementCompute, RoundStyle>,
    Sm90ScalarBroadcast<ElementScalar>,                            // beta
    Sm90SrcFetch<ElementC>,                                        // C
    Sm90EVT<Sm90Compute<multiplies, ElementCompute, ElementCompute, RoundStyle>,
      Sm90ScalarBroadcast<ElementScalar>,                          // alpha
      Sm90AccFetch>>;                                              // acc
```

2x 没有这一层：`DefaultGemmWithVisitor` 的 `FusionCallbacks` 模板参数直接就是那棵树，没有
标签、没有预置操作库。所以在 Ampere 上想要 `alpha*acc + beta*C` 要么用老的
`LinearCombination`（非 EVT），要么自己拼树；在 Hopper 上可以直接给标签。

一个 3x 特有的节点值得点出：`Sm90SrcFetch`（`sm90_visitor_load...hpp:91`）。它取的是
**collective epilogue 已经通过 TMA 加载好的 C 矩阵**，不是 `Sm90AuxLoad` 那种自己发起
加载的额外张量。2x 没有对应物——2x 路径下 C 也得当 aux tensor 自己 load（本仓库
`evt_attention_scores.h` 里 `ptr_C = nullptr`、bias 走 `VisitorAuxLoad`，就是这个原因）。

### 2x 与 3x 对照小结

| | 2x（Sm70–Sm89） | 3x（Sm90+） |
| --- | --- | --- |
| 命名空间 | `epilogue::threadblock` | `epilogue::fusion` |
| 树别名 | `Sm80EVT` = `TreeVisitor2x` | `Sm90EVT` = `Sm90TreeVisitor` |
| DAG | `Sm80TopologicalVisitor` | `Sm90TopologicalVisitor` + `Sm90SplitTreeVisitor` |
| visit 坐标 | `(iter_idx, row_idx, column_idx, frg_idx)` | `(epi_v, epi_m, epi_n)` |
| 回调链 | 单链 7 钩子 | producer 3 钩子 + consumer 8 钩子 |
| 归约 | 只有 atomic | atomic 或 workspace + last-CTA final |
| 能改已算结果 | 不能 | 能（`reduce` 拿到 `visit_results`） |
| C 矩阵 | 当 aux tensor 自己 load | `Sm90SrcFetch` 复用 collective 的 TMA |
| 预置融合库 | 无 | `operations.hpp` + `FusionCallbacks` 特化 |
| 输出流水 | smem ping-pong（`Stages<=2`） | TMA store pipeline |

要点：**抽象是同一个，能力不完全相同**。3x 的 `reduce` + `visit_results` 是真正多出来的
表达力；其余差异是架构适配。

### Python EVT 编译器（前端）

论文第 5.2 节的编译器在 `3rdparty/cutlass/python/cutlass_cppgen/backend/evt/`：

```text
frontend/python_ast.py      从 Python 函数 trace 出 DAG IR
ir/                         dag_ir.py + {load,compute,store,layout}_nodes.py（四类节点）
passes/                     pass 流水（见下）
backend/sm80_emitter.py     emit Sm80EVT / Sm80TopologicalVisitor 的 C++
backend/sm90_emitter.py     emit Sm90EVT / Sm90TopologicalVisitor 的 C++
backend/sm100_emitter.py    Blackwell
```

pass 顺序写在 `frontend/frontend_base.py:79-90`：

```text
PassPreprocessRed              → 归约预处理
PassGetArgumentType            → 生成 ctypes Arguments 结构
PassShapeTypePropagation       → 论文 "Shape Type Propagation"（两阶段：填缺失 / 消隐式广播）
PassLayoutManipulateElimination→ 论文 "Layout Node Elimination"（把 reshape/permute 推进 load/store）
PassGetImpl                    → 论文 "Get Implementation"（按 stride 选实现）
PassDAG2Tree                   → 把非树子图合成 TopoVisitorNode
PassFixElementD                → 修正输出 dtype
```

和论文 Fig. 10 的框图一一对应。`PassDAG2Tree`（`passes/pass_dag_2_tree.py`）的做法是：找出
出度 > 1 的节点，对它的所有 user 求最低公共祖先（LCA），把 LCA 之下的那块子图打包成一个
`TopoVisitorNode`——即**只在必须的地方用 topological visitor，其余仍用 tree visitor**。
这是论文 Fig. 8(D)/(E) 两类 visitor 的取舍在实现里的落点：topological visitor 要临时寄存器
buffer，能少用就少用。

`epilogue.py:59` 的 `EpilogueFunctorVisitor(cc, visitor, element_compute)` 里
`self._epilogue_stages = 1 if cc == 80 else None`——就是 2x 那个 `Stages <= 2` 断言的上层
体现。

## 遗留问题

- `Sm90SplitTreeVisitor` 我只读了 `visit` 的转发逻辑，没确认它在 `reduce` / `postreduce`
  上如何处理多个 output tree 的顺序。
- `Sm90ColReduction` 的非 atomic 路径我读了 counter + final 的骨架，没逐行核对
  `reduction_buffer` 的 layout 与 workspace 大小计算。
- `sm100_*` / `sm120_*` 的 callbacks 只看了文件清单，没读；Blackwell 上是否又有新钩子未确认。
- `FusionCallbacksTraits` 的通用版本把 `ElementCompute` 设成 `void`（`callbacks.hpp:66`），
  自定义 EVT 走路线 B 时哪些 collective 特性会因此不可用，我没查。

## Profile

<!-- 本工作区不新增 kernel；本仓库硬件是 SM89，3x 路径无法在本机实测。 -->

## 后记

<!-- HUMAN: 读完补充自己的结论、踩坑和下一步。 -->
