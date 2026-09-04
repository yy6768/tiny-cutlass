# EVT 学习记（零）：Epilogue Visitor Tree 到底是什么

## 前言

起因是上一轮在 swin 工作区调研「EVT 能不能融合 window attention」，结论是只能融合
`scores = scale·QK^T + bias` 这一段，softmax 和 P·V 都进不去。那次是拿结论用，没搞清楚
EVT 本身是什么：它是一篇 ASPLOS'24 的论文（一整套编译器），还是 CUTLASS 头文件里的一组
模板？两者是什么关系？为什么 `Sm80EVT` 是个 `TreeVisitor2x` 的 alias，而 Hopper 那边
又是另一套 `Sm90TreeVisitor`？

所以这次要把它当成一个技术本身读一遍：

1. 原论文讲了什么（要把 PDF 下下来，翻译一遍）；
2. CUTLASS `include/` 里 EVT 具体是怎么实现的；
3. 结合具体案例看它怎么用；
4. 最后回到 SwinAttention：如果要用 EVT 做融合，设计应该长什么样（只要设计，不写代码）。

目前不清楚的地方（进正文前先记下来）：

- 论文里那套 graph pass + ILP partitioner，和 CUTLASS 仓库里的模板，边界在哪；
- `Sm80EVT`（2x epilogue）和 `Sm90EVT`（3x TMA warp-specialized epilogue）除了架构，
  抽象上差在哪；
- EVT 的 reduction 节点（`VisitorColReduction` / `VisitorRowReduction`）既然存在，
  为什么 softmax 还是融不进去。

起点材料：

- 论文：`csrc/evt/papers/evt-asplos24.pdf`（ACM DL，`10.1145/3620666.3651369`）；
- Sm80 路径：`3rdparty/cutlass/include/cutlass/epilogue/threadblock/fusion/`
  （`visitor_2x.hpp` / `visitor_load.hpp` / `visitor_store.hpp` / `visitor_compute.hpp`）
  + `threadblock/epilogue_with_visitor_callbacks.h`
  + `gemm/kernel/{default_,}gemm_universal_with_visitor.h`；
- Sm90 路径：`3rdparty/cutlass/include/cutlass/epilogue/fusion/sm90_visitor_*.hpp`；
- 编译器前端：`3rdparty/cutlass/python/cutlass_cppgen/backend/evt/`；
- 案例：examples `15`（sparse + visitor）、`47`（StreamK + broadcast）、`49`（Hopper
  CustomEVT）、`35`/`37`（visitor 的前身，GEMM+softmax / GEMM+layernorm）；
- 本仓库已验证的 EVT 实例：`csrc/swin/kernel/evt_attention_scores.h`。

## Overview

### 一句话

EVT 是**把 GEMM/conv 的 epilogue 从「一个整体 functor」改成「一棵可组合的算子树」**的
技术：树的叶子是 load 类节点（取 accumulator、取标量/行/列/整块 aux tensor），内部节点
是 elementwise compute，根是 store 类节点（写回或归约写回）；每个节点独立实现、独立
优化，由 tree visitor 在编译期组装成一个 epilogue 类。

从抽象上说它做的是这件事（论文 Fig. 7/8 的核心）：

```text
Mainloop-Epilogue 抽象：kernel = Mainloop × Epilogue，支持 |mainloop| × |epilogue| 种组合
                        问题：epilogue 是"一个算子"，consumer 组合数爆炸，写不完

Epilogue Visitor  ：把 consumer 建模成计算图，节点集小（load/compute/store/layout 四类），
                    组合的爆炸发生在边集上 → 只实现节点，由编译期把边接起来
```

### 两层东西，别混

这是我一开始混掉的点，先钉住：

| 层 | 内容 | 在哪 |
| --- | --- | --- |
| 论文的 EVT 系统 | joint fwd-bwd graph pass（loss elimination / decomposition / reduction elimination）+ ILP partitioner + EVT operator compiler | 上游 `apuaaChen/EVT_AE`，**不在** CUTLASS |
| EVT 的 CUDA 抽象 | visitor 节点库 + tree/topological visitor + 驱动它的 epilogue 和 kernel | CUTLASS `include/`（本文重点） |
| EVT 的 Python 前端 | 从 Python function/DAG 生成上面那些模板的 C++ 代码 | CUTLASS `python/cutlass_cppgen/backend/evt/` |

论文摘要里 "The CUDA templates of EVT are actively maintained under the official CUTLASS
repository" 就是指第二层。所以「EVT 是什么」的答案要分开答：作为论文它是一个训练图编译
器；作为 CUTLASS 里能用的东西，它是第二层那套 visitor 模板。

### CUTLASS 里的两条实现路径

同一个抽象，两套代码，别互相套用：

```text
2x 路径（Sm70~Sm89，Ampere/Ada 用这条）
  cutlass::epilogue::threadblock::
    Sm80EVT<NodeOp, ChildOps...>            = TreeVisitor2x            visitor_2x.hpp:317
    Sm80TopologicalVisitor<...>             = TopologicalVisitor2x     visitor_2x.hpp:324
    VisitorAccFetch / VisitorScalarBroadcast / VisitorRowBroadcast /
    VisitorColBroadcast / VisitorAuxLoad                               visitor_load.hpp
    VisitorCompute                                                     visitor_compute.hpp
    VisitorAuxStore / VisitorColReduction / VisitorRowReduction /
    VisitorScalarReduction                                             visitor_store.hpp
  ↑ 被 EpilogueWithVisitorCallbacks 驱动     epilogue_with_visitor_callbacks.h:65
  ↑ 被 GemmWithEpilogueVisitor 调用          gemm_universal_with_visitor.h:55
  ↑ 由 DefaultGemmWithVisitor 装配           default_gemm_universal_with_visitor.h:102

3x 路径（Sm90+，TMA warp-specialized）
  cutlass::epilogue::fusion::
    Sm90EVT<NodeOp, ChildOps...>            = Sm90TreeVisitor          sm90_callbacks...hpp:58
    Sm90SplitTreeVisitor / Sm90TopologicalVisitor                      sm90_visitor_tma...hpp:633/681
    Sm90AccFetch / Sm90AuxLoad / Sm90{Scalar,Row,Col}Broadcast         sm90_visitor_load...hpp
    Sm90Compute                                                        sm90_visitor_compute...hpp
    Sm90AuxStore / Sm90{Scalar,Row,Col}Reduction                       sm90_visitor_store...hpp
    Sm90TopKSoftmaxColReduction                                        sm90_visitor_topk_softmax.hpp:335
  ↑ 被 CollectiveEpilogue(sm90_epilogue_tma_warpspecialized) 驱动
```

两条路径的**回调协议不同**，这是本文要讲清的一个重点：

```text
2x : begin_epilogue -> [begin_step -> [begin_row -> visit... -> end_row] -> end_step] -> end_epilogue
3x : begin -> [begin_loop -> previsit -> visit... -> reduce -> postreduce -> end_loop] -> end
```

2x 的钩子名字直接对应论文 Fig. 9 的 `begin_xxx` / `visit` / `end_xxx`；3x 因为 epilogue
本身是 TMA + smem pipeline，多了 `previsit` / `reduce` / `postreduce` 三个位置。

### 数据流：一次 epilogue 里发生什么

以 2x 为例（`epilogue_with_visitor_callbacks.h`）：

```text
mainloop 结束，accumulator 在寄存器（warp-level 分布，不是输出 layout）
  ↓ AccumulatorFragmentIterator + warp_tile_iterator_ 写 smem      （换 layout）
  ↓ __syncthreads
  ↓ SharedLoadIterator 读回寄存器                                   （按输出 thread map）
  ↓ 对每个 fragment 调 callbacks.visit(iter, row, col, frg_idx, frg_acc)
       └─ tree visitor 递归：先算所有 child，把结果作为额外参数喂给 node op
  ↓ 根节点（AuxStore）在 end_step 里 global_store
```

关键在于 **visit 的参数只有 `(iter_idx, row_idx, column_idx, frg_idx, frg_acc)`**——一个
线程看到的是自己那几个元素，加上「我在 CTA 输出 tile 里的哪一行哪一列」。这就直接决定了
EVT 的能力边界：per-element 和「沿 tile 内某个轴」的归约能做，跨 CTA tile 的归约只能靠
atomic（`VisitorColReduction` 的 `end_row` 里就是 `atomic_reduce`），而需要「先看完整行
再回头改这一行」的 softmax 做不了。

### 正文阅读顺序

1. `01-paper-translation.md` — 原论文全文译文（含 Epilogue Visitor 抽象、operator
   compiler、graph pass、ILP partitioner）。译文里 CUTLASS 无关的部分标注清楚。
2. `02-cutlass-2x-implementation.md` — Sm80 路径源码：`TreeVisitor2x` 怎么递归、
   `VisitorImpl2x` 的 callbacks tuple、四类节点各自填哪些钩子、
   `EpilogueWithVisitorCallbacks` 的两级 smem pipeline、`DefaultGemmWithVisitor` 的装配。
3. `03-cutlass-3x-implementation.md` — Sm90 路径：`Sm90TreeVisitor` 与 2x 的差异、
   producer/consumer 双回调、`reduce`/`postreduce` 钩子、`FusionCallbacks` 与
   `CollectiveBuilder` 的关系、`Sm90TopKSoftmax` 为什么能做「一部分 softmax」。
4. `04-case-studies.md` — 具体案例：example 47（bias broadcast + 两个 aux load + StreamK）、
   example 15（sparse）、example 49（Hopper CustomEVT）、example 35/37（visitor 前身，
   GEMM+softmax / +layernorm 为什么是多 kernel），以及本仓库
   `csrc/swin/kernel/evt_attention_scores.h` 这个已验证实例的逐行读法。
5. `05-swin-attention-evt-design.md` — SwinAttention 用 EVT 做融合的**设计**：能融什么、
   融不了什么、几种可行切分方案及其取舍。只写设计，不写代码。

### 本文边界

- 不讲 mainloop（TensorOp MMA、cp.async pipeline、TMA），只讲 epilogue 之后的事。
- 不讲论文里 ILP partitioner 的实现细节到可复现的程度，只讲它要解决什么、约束怎么编码。
- Python EVT 前端只讲 pass 流水的作用，不逐个 pass 读代码。
- 没有 Profile 数据：本工作区不新增 kernel，`Profile` 一节留空，EVT 实测数据属于
  `csrc/swin` 那条链路，等它 verify + bench 通过后再写。

## Profile

<!-- 本工作区不新增 kernel，无实测数据。EVT scores 核的 NCU 数据归属 csrc/swin。 -->

## 后记

<!-- HUMAN: 读完补充自己的结论、踩坑和下一步。 -->
