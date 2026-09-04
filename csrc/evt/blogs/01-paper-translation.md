# EVT 学习记（一）：原论文译文

## 前言

这一篇是 EVT 原论文的中文译文，方便后面几篇直接引用论文的术语和图号。

- 出处：Zhaodong Chen, Andrew Kerr, Richard Cai, Jack Kosaian, Haicheng Wu, Yufei Ding,
  Yuan Xie. *EVT: Accelerating Deep Learning Training with Epilogue Visitor Tree*.
  ASPLOS '24, April 27–May 1, 2024, La Jolla, CA, USA. DOI `10.1145/3620666.3651369`。
  Creative Commons Attribution International 4.0 License。
- 本地 PDF：`csrc/evt/papers/evt-asplos24.pdf`（4.1 MB，16 页）；
  纯文本抽取：`csrc/evt/papers/evt-asplos24.txt`（`pdftotext -layout`）。

翻译口径：

- 按节翻译，保留论文的节号、图号（Figure）、表号（Table）、公式号，方便对照原文；
- 术语第一次出现时给英文（如「主循环 mainloop」），后面沿用中文；
- 公式在纯文本抽取里丢了排版，译文按 PDF 页面重排，标注了页码；
- 论文里描述的 graph pass 与 ILP partitioner 属于上游 `apuaaChen/EVT_AE` 项目，
  **不是** CUTLASS `include/` 的内容——译文照译，但在需要的地方加了「译注」区分。

译注一律用 `> 译注：` 标出，不与原文混排。

## 摘要

随着深度学习模型日益复杂，深度学习编译器对于提升系统效率、发掘隐藏的优化机会变得至关
重要。尽管在推理负载上已经取得了很好的加速，现有编译器在训练负载上仍面临显著局限。第一，
训练计算图包含难以融合的复杂算子，例如归一化、损失函数和归约，这限制了 kernel 融合这类
优化机会。第二，训练图中连接前向与反向算子的额外边，使得寻找最优且可行的融合划分变得
困难。更重要的是，现有编译器要么无法在现代 GPU 上生成达到 state-of-the-art 性能的
kernel，要么无法容纳多样的融合模式。

本文提出 Epilogue Visitor Tree（EVT），一个克服上述局限的新编译器。EVT 采用新的图级
编译 pass 来发掘隐藏的融合与优化机会；引入基于整数线性规划（ILP）的划分器，在复杂的
前向-反向联合图中高效求解最优且可行的划分；并提出 Epilogue Visitor 抽象与 EVT 算子
编译器，自动生成灵活的 epilogue，且能与 CUTLASS 等 SOTA 库的高性能主循环实现集成。EVT
在跨领域的多种训练负载上评估，取得 1.26–3.1× 的加速。

> 译注：摘要脚注里明确写了「EVT 的 CUDA 模板在官方 CUTLASS 仓库中持续维护」
> （`github.com/NVIDIA/cutlass`），复现代码在 `github.com/apuaaChen/EVT_AE`。这正是
> 「论文的 EVT 系统」和「CUTLASS 里的 EVT 模板」两层的分界。

## 1 引言

深度学习的快速演进带来了模型复杂度与数量的激增。这些模型通常用 PyTorch 这类面向灵活
编程的框架实现，因而无法高效利用 GPU 等硬件提供的资源，性能次优。手工优化这些模型需要
大量工程投入与专业知识，即使专家也可能忽略复杂模型中隐藏的优化机会。

深度学习（DL）编译器在应对这一挑战中扮演关键角色。典型的 DL 编译器通过三个主要步骤优化
用户定义的模型：图级优化、划分（partition）、算子级优化。具体来说，输入模型首先被 trace
成图表示，节点表示算子，边表示数据流；图级优化通过常量折叠、公共子表达式消除等手段简化
图；随后划分器把图切成若干划分，每个划分再由算子级优化（例如用算子编译器生成的融合
kernel 替换）进一步优化。

尽管在推理效率优化上已有可观进展，训练负载的优化仍存在三个关键局限。

**局限 1.** 现有算子编译器无法在容纳多样融合模式的同时生成 state-of-the-art 性能的融合
kernel。以 TVM 为代表的基于循环（loop-based）的编译器把算子表示成嵌套循环，难以抽象出
专用优化，也难以为 GEMM、卷积这类核心算子利用现代 GPU 的加速器特性与数据通路。另一方面，
利用专家开发的模板库的模板元编程（TMP）编译器缺乏支持训练图中多样融合模式的灵活性。

**局限 2.** 现有研究关注前向与反向传播，很少有针对损失函数的优化。然而在诸如极端分类
（extreme classification）等场景中，损失函数及其反向传播会成为性能瓶颈。

**局限 3.** 现有划分算法无法在训练图中找到可行且最优的划分。与推理不同——推理是一串
算子，用简单启发式就能划分——训练图中还有把保存的激活传给反向算子以计算梯度的额外边。
启发式划分很容易产生含环的、不可行的结果；而任意地把前向图与反向图分开，又会掩盖跨前向-
反向融合算子的机会。此外，现有划分算法是为特定算子编译器定制的，缺乏集成额外优化目标、
启发式与约束的可扩展性。

### 1.1 我们的方法

我们提出 Epilogue Visitor Tree（EVT），一个针对上述所有局限设计的新编译器。

本文的核心是 **Epilogue Visitor Tree 抽象**，用于解决局限 1。关键想法是：像 TMP 编译器
那样通过「主循环-epilogue 融合」组装融合 kernel，同时通过 **visitor 设计**解决 TMP
编译器中 epilogue 的可扩展性问题——epilogue 算子可以被单独开发、优化和验证，然后由编译器
在树结构下组装起来。在这一抽象下，专家设计的主循环保证了 state-of-the-art 性能（所有用
CUDA 写的专用优化都被自动纳入），而编译器组装的 epilogue 提供了处理训练中多样融合模式的
强灵活性。

尽管 Epilogue Visitor Tree 抽象提供了很强的灵活性，它仍然对可融合的模式施加约束。例如，
只有被支持的 epilogue 算子才能融合；归约与其 consumer 在 epilogue 内的融合也很困难，
有时并不划算。这些约束由协同设计的图级优化与划分算法缓解。

图 1 给出 EVT 的总体结构。它直接接收 PyTorch 等框架的模型，转换为前向-反向联合图，其中
训练样本与模型参数是输入，梯度是输出。

**图级优化。** 为解决局限 2，EVT 通过「损失消除」（Loss Elimination）优化损失函数及其
反向传播。这一做法通过认识到「损失值本身并不参与反向计算」，打开了融合损失函数及其反向
传播的机会。

为缓解 EVT 抽象的约束，采用「分解」（Decomposition）把 `batch_norm` 这类复杂的、不可融合
的算子拆成 EVT 抽象下可融合的算子。此外，「归约消除」（Reduction Elimination）通过识别
归约结果可以等价表达为一串可融合算子的情形，缓解归约融合的约束。

**划分。** 局限 3 由我们的 ILP 划分器解决。它把划分建模成 ILP 问题以求出可行的全局最优
解。这一形式化还允许集成额外的优化目标、启发式与约束——这种灵活性使得图级优化未能解决的
EVT 抽象约束可以编码为 ILP 求解器的约束，从而保证求出的所有划分都可融合。

由于无环划分问题是 NP-Hard 的，我们提出若干简化技术来降低复杂度，使含数百万决策变量的
大规模神经网络能在数分钟内求解。

**算子编译器。** 每个含 GEMM 这类重算子的划分会被派发给我们的 EVT 算子编译器。它足够
灵活以支持训练中多样的融合模式，同时生成达到 SOTA 性能的融合 kernel。

最后，EVT 返回一个可部署模型，与 CUDA graph 等与本工作正交的优化技术兼容。EVT 可以作为
`torch.compile` 的后端，用一行代码优化模型。

## 2 相关工作

### 2.1 图级优化

现有 DL 编译器采用了各种图级优化。TVM 用常量折叠、分解、数据布局变换等 pass 优化它的
图级 IR relay。

然而针对损失函数及其反向传播的优化很少，而它在很多情况下会是性能瓶颈。此外，现有图级
优化（例如分解）通常独立于底层算子级优化设计，错过了在图级解决算子级优化某些局限的机会。

### 2.2 划分算法

多数现有研究在划分计算图时采用简单启发式。例如 TVM 把算子分成四类：injective（逐元素）、
reduction（归约）、complex out-fusible（可以把逐元素 map 融合到输出）、opaque（不可融合），
并定义诸如「归约可以与输入侧的 injective 算子融合」「GEMM 这类 complex out-fusible 节点
可以把逐元素算子融合到输出」这类启发式规则。然而正如第 4 节将说明的，这些启发式会导致含
环的不可行划分或次优解。此外，现有划分算法缺乏纳入额外优化目标、启发式与约束的灵活性。

深度学习系统社区之外也有关于无环划分问题的研究。我们的 ILP 划分器受 Nossack 与 Pesch 的
启发，他们把无环划分问题形式化为 ILP。基于 ILP 的形式化在通过 ILP 约束编码新约束与启发式
方面提供了很好的可扩展性。然而 Nossack 与 Pesch 方法的高复杂度使其无法直接用于求解含
数千节点的大规模神经网络，而且当引入额外约束时，他们从 Kernighan 解得到的下界不再有效。
在第 4 节中，我们提出一种把计算图切成更小图的新方法，使其能在合理时间内求解。

### 2.3 算子级优化

kernel 融合是算子级优化中的关键优化。融合时，处于生产者-消费者关系的多个算子被融合成
一个，中间结果可以直接缓存在寄存器或 shared memory 中以减少内存访问。融合 kernel 的性能
由每个算子实现的质量决定，而灵活性由算子编译器对齐不同算子循环结构的能力决定。

AITemplate、Bolt 这类模板元编程（TMP）编译器使用 CUTLASS 等专家开发的模板库来达到最优
性能。然而现有 TMP 编译器缺乏覆盖训练负载中多样融合模式的灵活性，它们只支持模板库定义的
有限模式集，可扩展性受限。

TVM、Tiramisu、Nvfuser、Tensor Comprehensions 这类基于循环的编译器把运算表示为循环，
并施加 loop fission、fusion、parallel、vectorization 等 schedule 把它们映射到 GPU。基于
循环的编译器在推理任务上取得了显著成功，甚至超过专家精心打造的 SOTA 库。基于循环的抽象
能轻松地把小问题规模的负载切分到足够多的 threadblock，解决 GPU 流多处理器（SM）利用率
低的问题。

相比之下，训练负载有足够大的 batch size 来占满 SM，kernel 性能受计算与内存利用率约束。
提升这些利用率需要用 CUDA 做专用的、有时是逐案例的优化，而这些很难表达为 IR 上的变换。

因此，PyTorch 的 inductor 等现有方法尽管具备生成 kernel 的能力，对 GEMM 这类重算子仍然
依赖 cuDNN 的 kernel 或 CUTLASS 的模板。

## 3 图级优化

本节先讨论采用的主要优化，然后给出一个运行示例，说明这些优化如何简化计算图（特别是损失
函数及其反向传播）并暴露新的融合机会。

### 3.1 优化

**损失消除（Loss Elimination）。** 尽管听起来反直觉，我们发现损失值可以作为死节点从
前向-反向联合图中消除，原因有两点：反向传播可以在不计算损失值的情况下完成；以及损失值
并非每次迭代都需要。

关于第一点，损失值不参与反向传播的计算。例如在广泛使用的 softmax 交叉熵损失函数中，
batch size 为 n 时，logits X 在 target Y 下的梯度可以化简为

```text
∂L/∂X = -(1/n)·OneHot(Y) + (1/n)·Softmax(X)                        (1)
```

有了这个式子，梯度可以在不需要损失值的情况下直接计算，且数值稳定性更高。

关于第二点，损失值只在用户需要观察它以评估训练过程时才需要。然而训练过程可能有数百万次
迭代，没必要每次迭代都检查损失。因此我们可以只在需要时计算损失，其余迭代使用加速版本。

基于这一领域知识，我们提出损失消除，把损失值作为死节点消除。但 PyTorch 等现有 DNN 框架
仍用损失值来触发反向传播，因此我们的实现方式是简单地把损失值替换为一个常量标量张量。
在底层，损失消除绕开了计算标量损失的那个归约。它把损失函数的前向和反向都变成逐元素运算，
从而创造新的融合机会并改善数值稳定性。

**分解（Decomposition）。** 深度学习框架包含许多不可融合的运算，例如 `logSoftmax`、
`addmm`，它们的设计目的是简化编程并利用已有的融合 kernel。然而这些运算也成为 kernel 融合
的主要障碍，因为每一个都可能需要专门的 epilogue 算子来处理。为解决这一问题，我们为这些
运算注册分解规则，通过模式匹配与重写把它们拆成逐元素运算和归约这类基本可融合运算。
表 1 给出了这些分解规则的例子。

**表 1：EVT 中的分解示例**（PDF p.304）

| 模式 | 分解 |
| --- | --- |
| `logSoftmax(X)` | `log(Softmax(X))` |
| `nllLossBp(X, Y)` | `-(1/n)·OneHot(Y)` |
| `logSoftmaxBp(dY, X)` | `dY - logSoftmax(X)·Σ dY` |
| `dropout(X, p)` | `X ⊙ (rand_like(X) > p) / (1-p)` |
| `dropoutBp(dY, M, p)` | `dY ⊙ M / (1-p)` |
| `reluBP(dY, X)` | `dY ⊙ (X >= 0)` |
| `addmm(X, W, b)` | `mm(X, W) + b` |
| `BN(X, α, β)` | `μ = ΣX/n`, `m2 = ΣX²/n`, `σ = sqrt(m2 - μ²)`, `(γ/σ)(X-μ) + β` |
| `BNBp(dY, X, μ, σ, γ)` | `dβ = ΣdY`, `X̂ = (X-μ)/σ`, `dγ = Σ(dY ⊙ X̂)`, `dX = (γ/σ)(dY - dβ/n) - dγ·γ·X̂/(n·σ)` |

这些分解规则与已有研究的一处不同是：它们的设计考虑了 EVT 算子编译器的融合约束（5.2 节）。
一方面，分解结果尽可能由 EVT 算子编译器支持的算子组成；另一方面，尽量避免对归约结果的
依赖。例如受 Jung 等人（2019）启发的 Batch Normalization（BN）分解，利用
`σ = sqrt(ΣX²/n - (ΣX/n)²)` 来解耦两个归约，从而获得更多融合机会。

除了这些注册的分解规则，我们还提供了一个简单接口，让用户以普通 Python 函数声明 pattern
与 replacement 来注册自定义规则：

```python
def pattern(bias, x, weight):
    return torch.ops.aten.addmm(bias, x, weight)

def replacement(bias, x, weight):
    mm = torch.ops.aten.mm(x, weight)
    return torch.ops.aten.add(mm, bias)
```

**归约消除（Reduction Elimination）。** 与现有算子编译器类似，我们的 EVT 算子编译器在
把归约与其 consumer 融合时面临困难，从而限制了融合机会。这一限制源于归约需要特殊的
并行化——要求所有待归约元素被划分到同一个 threadblock，而这种并行化通常与 GEMM 等算子的
主循环 tiling 不兼容。

> 译注：这一句是整篇论文里对「为什么 softmax 这类跨轴归约融不进 epilogue」最直接的解释，
> 后面 `05` 篇讨论 Swin attention 时会反复用到它。

在某些场景下，当归约涉及具有常量归约值的算子时，归约运算可以被变换成一串可融合的逐元素
算子。例如，给定整数 `y_i ∈ [0, n)`（`0 ≤ i ≤ m`）和标量 α，有

```text
Σ_{j=0..n-1} ( α·OneHot(y) )_{:,j} = α                             (2)
```

我们引入归约消除，自动识别这类机会并重写图以打开额外的融合机会。一个主要挑战是：具有
常量归约值的运算并不总是归约的直接输入，它们之间可能存在一串可结合的算术运算（+、-、×、÷）。
沿用表达式代数重结合的既有方法，我们用 3 步解决：

- 抽取产生该归约的子图，只包含与该归约可结合的算子。遇到不可结合的节点时，用一个
  placeholder 替换它，并在它具有常量归约值时加以标注。
- 按 Briggs 等人（1994）的方法做表达式重结合，给归约节点分配 rank -1，使其 rank 最低。
  重结合之后，归约总是以 placeholder 作为输入。如果该 placeholder 具有常量归约值，就把
  它折叠成一个静态张量。
- 做常量传播；如果子图中不再含归约节点，把它代回原图。

**其他图简化。** 除上述专门优化外，我们还加入了一组标准图级优化来进一步简化计算图，包括
带表达式重结合的常量传播、公因子提取、公共子表达式消除、死代码消除等常规技术。

### 3.2 示例

为说明上述优化如何简化计算图并揭示新的融合机会，这里给出一个运行示例，展示
`softmax_cross_entropy` 及其反向运算如何被化简为单个融合的 softmax，既提升数值稳定性
也改善整体性能。

`softmax_cross_entropy` 是社区广泛采用的标准损失函数，在很多情况下会成为性能瓶颈。例如
我们的 profiling 显示，在 ogbn-mag 数据集上训练图卷积网络（GCN）时，损失及其反向传播占
总迭代时间的 13% 以上。低效来自沿 batch 维的大归约，它对混合精度训练不友好（数值稳定性
问题）。

图 2 展示了 trace 得到的计算图，其中含四个不可融合节点。经过表 1 的模式分解后，这些不可
融合节点被拆成基本可融合节点。接着我们的损失消除把 `nllLoss` 作为死节点消除。而从
`logSoftmax BP` 分解出的那个节点原本阻碍与 `nllLossBp` 分解出的算子融合，我们的归约消除
pass 把它折叠成基本逐元素运算。最后，其他图简化 pass 清理计算图，最终使它能被我们的算子
编译器融合。由于优化后的计算图不含归约，所有中间结果以全精度保存在寄存器中，它的数值
稳定性比原版本好得多。

## 4 划分器

图优化完成后，下一步是把它切成互不相交的划分以便融合。我们先用图 3 的简单例子说明现有
研究使用的启发式可能无法找到可行且最优的划分。

**例子。** 图 3 展示了两层的前向-反向联合图。逐元素算子标绿，`mm` 表示可以把逐元素算子
融合到输出的矩阵乘。就可行性而言，`{mm2, tanh, dtanh}` 在 TVM 的规则下是合法的，但由于
成环而不可行。就最优性而言，虽然 `{mm1, relu}` 与 `{mm5, ">0", ×}` 构成合法划分，但它是
次优的——划分 `{mm1, relu, >0}` 与 `{mm5, ×}` 能省下更多内存访问，因为 `">0"` 的输出可以
用 bitmask 存储。

**我们的方法。** 我们提出一种把划分形式化为整数线性规划问题的新算法，它有三个关键好处：

- **可行性**：包括无环在内的要求被编码为 ILP 约束；
- **最优性**：目标函数最大化每个划分内通过融合节省的内存访问；
- **可扩展性**：额外要求与启发式可以轻松编码为约束。

有了「可扩展性」，划分器可以轻松纳入来自 EVT 抽象的约束与启发式，以及自定义目标。

本节先按 Nossack 与 Pesch 的方法讨论 ILP 形式化，然后给出我们缩短求解时间的新技术。

### 4.1 整数线性规划形式化

设 `G = (V, E)` 为拓扑排序后的计算图，节点集 `V = {1, ..., n}`，边集
`E ⊆ {(i, j) | i, j ∈ V}`。目标是在若干约束与启发式下，找到 V 的互不相交的划分
`{V_1, ..., V_K}`，使每个划分都可融合。收益是每个划分内部边引起的内存访问之和。因此我们
给每条边分配边权 `w_ij ∈ N+`，表示沿该边传递的张量大小。沿用 Nossack 等人的做法，定义
四类决策变量：

- `x_ik ∈ {0,1}`：`x_ik = 1` 表示节点 i 在簇 k 中；
- `y_ijk ∈ {0,1}`：`y_ijk = 1` 表示节点 i、j 同属簇 k；
- `z_kl ∈ {0,1}`：`z_kl = 1` 表示簇 k 与簇 l 的节点之间存在边；
- `u_k ∈ Z`：辅助变量，用于形式化保证无环划分的 Miller-Tucker-Zemlin（MTZ）子回路消除
  约束。

ILP 问题形式化如下（PDF p.306 公式 3a–3j）：目标函数 (3a) 最大化每个划分内部的边权之和；
约束 (3b) 保证每个节点被唯一地分到某个划分；约束 (3c) 与 (3d) 编码额外的约束与启发式
（不能融合 / 必须融合）；约束 (3e)、(3f) 连接决策变量 x、y、z；约束 (3g) 形式化无环约束；
约束 (3h) 减少解的对称性；最后 (3i)、(3j) 是各决策变量的取值范围。

> 译注：公式 3a–3j 的具体符号在纯文本抽取中丢失了下标排版，此处只译各式的作用；需要
> 精确形式请看 PDF p.306。

### 4.2 降低复杂度

尽管形式化 (3) 提供了可行性、最优性与高可扩展性，直接求解它是不实际的：四类变量引入
`O(n³)` 个决策变量，而分支定界法在最坏情况下需要指数时间才能找到最优解。为解决这一挑战，
受「神经网络由堆叠相似层构成」这一事实启发，我们把 V 切成互不相交的分量，在每个分量内
分别求解式 (3)。此外我们缓存解，避免对结构相似的分量重复求解同一 ILP 问题。主要挑战是
保证各分量解的组合仍然可行且最优。我们给出两步方案：

- 在不损害最优性的前提下把节点集 V 切成互不相交的分量 `{V_1, ..., V_C}`；
- 在每个分量内重建边以保证解的可行性，然后求解式 (3)。

**切分节点集。** 我们用算法 1 切分节点集。我们的洞察是：在最优解中，只有通过可融合边
弱连通的节点才可能属于同一划分。因此我们剪掉 G 中所有不可融合边，把它切成弱连通分量。
图 4 显示这一做法把 G 切成多个节点很少的分量，只把分量 1 的划分留给 ILP 求解器。

算法 1 中识别出两类不可融合边。第一类（第 1 行）是直接连接两个不可融合节点的边，例如
`mm` 的入边。第二类（第 2–8 行）是那些一旦融合就会造成环的边。我们利用 Nossack 等人
定理 3.3 的一个必要条件：若 G 中存在从 i 到 j 的有向路径，且 i、j 属于不同划分，则所有
能从 j 到达的节点都不能与 i 同属一个划分。在图 4 中，`tanh` 与 `dtanh` 之间的边就是据此
剪掉的，因为 `tanh` 不能与 `mm3` 融合，而 `mm3` 能到达 `tanh`。

**重建边并求解 ILP。** 用算法 2，对每个分量重建边以在求解 ILP 前保证可行性。下面的
定理 4.1 保证其解总是可行的，并在一个易验证的条件下是最优的——该条件在我们对各种 DL 模型
的评估中总是成立。

**定理 4.1.** 当 `is_optimal` 为 true 时，算法 2 返回全局最优且可行的解；否则返回一个
可行解。（证明见补充材料。）

图 5 用图 4 的分量 1 作为示例。若某节点有来自该分量的入边，它是 descendant（例如
`mm2`）；若它有指向该分量的出边，它是 ancestor（例如 `dtanh`）。我们用 3 步构建边：

- 识别当前分量的所有 ancestor 与 descendant（第 5、6 行）；
- 把 G 中所有可融合的边替换为双向边（第 4、10 行）；
- 若修改后的图中任意 descendant 与 ancestor 之间存在路径，则添加一条从该 descendant 到
  该 ancestor 的边（第 11 行），例如图 5 中的 `mm2 → dtanh`。

直觉是：如果无环约束既不放松也不收紧，那么图 5 的解就是可行且最优的。换言之，「图 4 中
存在一个涉及分量 1 节点的环」是「图 5 中存在环」的充要条件。

ancestor 与 descendant 充当辅助节点。如果分量 1 内的某次融合造成了涉及分量外节点的环，
该环必然包含 descendant 与 ancestor 的配对。我们用它们作为边界，把环切成分量内的路径与
分量外的路径。条件于是变成：「图 4 中任意 descendant-ancestor 配对之间存在路径」是「图 5
中存在同源同目标的路径」的充要条件。

从 ancestor 到 descendant 的分量内路径，通过保留所有内部边来保证充要性。然而从
descendant 到 ancestor 的外部路径会受其他分量内融合的影响。具体来说，融合不会移除路径，
但会创造等价于「把被融合边变成双向」的新路径。例如在图 6(B) 中融合 a 与 b 会创造一条
从 c 经 (a,b) 到 d 的路径，等价于添加图中那条红色箭头。为保证可行性，当图 5 中任意
descendant 到 ancestor 之间可能存在路径时，我们就添加相应的边。图 6 展示了四种不同情形：
(A)、(B) 是应当添加边的情形——(A) 中存在自然路径；(B) 中 a、b 属于同一分量因而可能被融合，
从而创造路径。相反，图 6(D)、(E) 是不需要加边的情形——(D) 中 a、b 属于不同分量因而不能
融合；(E) 中虽然 a、b 属于同一分量，但融合它们会造成环因而不可行。这一方法保证了条件的
必要性；若「修改后的图中存在路径时，原图中总存在对应路径」成立，它也是充分的（即最优的）。
第 12–13 行检查这一条件并返回指示器 `is_optimal`。

**缓存解。** 我们把 ILP 解存进数据库，使结构同构的分量可以复用。具体地，键用
Weisfeiler-Lehman 图哈希构造，使同构分量具有相同哈希。

## 5 算子级优化

本节先解释我们算子编译器背后的关键抽象「Epilogue Visitor」。该抽象通过直接利用专家为
生产者部分设计的实现来融合处于生产者-消费者关系的算子，在保证最优性能的同时，在编译期
组装消费者部分以适配不同融合模式。在此抽象之上，我们给出 EVT 算子编译器，它自动生成
支持多样模式且具有 SOTA 性能的融合 kernel。

### 5.1 Epilogue Visitor 抽象

Epilogue Visitor 抽象源自现有模板库与 TMP 编译器广泛采用的「主循环-Epilogue」抽象。

**主循环-Epilogue 抽象及其局限。** 如图 7(A) 所示，主循环-epilogue 抽象融合处于生产者-
消费者关系的算子。它要求消费者具有与生产者相同的空间循环集合，这样就能通过把消费者语句
注入生产者空间循环的末尾来完成融合。随后如图 7(B) 所示，循环被重构以通过 tiling、
并行化、软件流水等技术达到最优性能。主循环-epilogue 抽象把生产者当作主循环、消费者当作
epilogue。如图 7(C) 所示，不同的主循环与 epilogue 作为不同的 C++ 类分别开发，而图 7(D)
中的通用 kernel 模板把主循环与 epilogue 作为模板参数。这种模块化方式允许在编译期组装
不同的主循环-epilogue 对，支持 `|mainloop| × |epilogue|` 种模式。

然而，主循环-epilogue 抽象的「Epilogue」不具备可扩展性，因为消费者算子的数量增长极快。
手工实现所有可能的 epilogue 是不实际的，优化和验证复杂 epilogue 也构成显著挑战。

**Epilogue Visitor 抽象。** 我们的 EVT 抽象保留主循环，并进一步模块化 epilogue 以解决
其可扩展性问题。如图 8(A, B) 所示，我们把消费者建模成一个计算图而非单个算子。我们的关键
洞察是：可能的消费者算子的快速增长发生在**边集**而非**节点集**上——不同的消费者是同一组
基本算子的不同组合。这一洞察促使我们维护一个彼此解耦的基本算子高性能实现库，并开发一个
编译器在编译期为不同消费者组装它们。详细解释如下。

**支持的模式。** EVT 支持任何可以写成完美嵌套循环（perfectly nested loop）的消费者，
即所有语句都位于生产者的空间循环内（图 8(A)）。这一结构约束已被证明是简化变换的关键，
并被 MLIR 的 linalg dialect 等许多 SOTA 编译器广泛采用。这一要求足够宽松，能覆盖神经
网络中的大多数场景，包括逐元素 load/store、广播、（原子）归约以及逐元素计算。

**用 visitor 解耦 epilogue 算子。** 图 8 展示了 epilogue 的完美嵌套循环表示，其中原始
函数可以被分解为一串 load、store 与 compute 算子。我们把它可视化为图 8(B) 的图。注意
store 被建模为一个**透传节点**（transparent node），它转发自己的输入。为解耦每个节点的
实现，我们引入「visitor」并把图重写为图 8(C)，其中所有节点都成为叶子。我们引入两类
visitor：图 8(D) 的 **tree visitor** 处理出度为 1 的非叶节点；图 8(E) 的
**topological visitor** 处理其他一般情形——它按拓扑序排序输入，并创建一个描述输入间依赖
的边列表；运行时，topological visitor 用临时寄存器 buffer 保存各输入节点的中间结果。

**Epilogue 算子的实现。** 图 9 展示了解耦后的算子如何被单独优化和实现。所有算子的原始
循环结构被重构成同一套嵌套循环以获得最优性能，其中 `parallel_for` 绑定到 GPU 线程层级、
`vector_for` 被向量化。为方便起见，我们把顺序循环起始处的代码块命名为 `begin_xxx`，
末尾的命名为 `end_xxx`，最内层顺序循环的代码块称为 `visit`。

尽管由于完美嵌套循环约束，所有算子都可以只填 `visit` 代码块实现，但利用其他代码块可以
获得更好性能。例如图 9(A) 中，逐元素 load 可以先在 `begin_step` 把一小块矩阵拷进寄存器，
然后在 `visit` 中访问它们，这保证了合并访存并最小化地址/谓词运算。图 9(B) 中，逐列归约
可以在 `begin_epilogue` 初始化一个临时 buffer 存放 `visit` 期间累积的部分结果，然后在
`end_epilogue` 做并行归约，最后原子地归约到 global memory。最后，如图 9 底部所示，每个
算子都可以实现为提供若干成员函数（定义非空代码块）的模板类，这些函数将在运行时被
visitor 调用。

> 译注：这一段直接对应 CUTLASS `include/cutlass/epilogue/threadblock/fusion/` 里的
> `VisitorAuxLoad::Callbacks::begin_step`（预取到寄存器）与
> `VisitorColReduction::Callbacks::{begin_row, visit, end_row}`（部分归约 + atomic），
> `02` 篇会逐个对照。

### 5.2 EVT 算子编译器

有了 Epilogue Visitor 抽象，我们提出新的算子编译器 Epilogue Visitor Tree（EVT）。
「Tree」来自图 8(C) 中变换后的图成为一棵以最后一个 visitor 为根的树这一事实。编译器概览
见图 10。

编译器的核心是图 11 所示的图中间表示（IR）。它把 epilogue 建模成一个图，每个节点表示
一个算子。节点分为四类：**load、compute、store、layout**。每个节点还有四个字段：
**name、shape、stride、dtype**。shape 与 stride 是遵循 CUTLASS cute 库的 tuple。例如
shape `(M, N)` 与 stride `(N, 1)` 表示按 row-major 存储的 M×N 矩阵；张量在内存中的存储
索引可以通过 shape 与 stride 的内积计算。

为支持不同输入格式，IR 由前端以最少的信息与约束构造。随后优化与 lowering schedule 的
pass 填充缺失信息，并把 IR 规范化以满足 epilogue visitor 抽象。C++ emitter 接着遍历 IR
并 emit epilogue 的模板类。这一设计使得为其他编译工作流添加新前端变得容易，同时复用其余
逻辑。

**前端（Frontend）。** EVT 支持从上一节生成的划分到 Python 函数等多种用户输入。Python
前端的例子如下，它定义了图 11(A) 的 epilogue：

```python
def example_epilogue(accum, bias):
    add = reshape(accum, new_shape=(128, 4, 64)) + bias
    permute_1 = permute(add, indices=(2, 1, 0))
    reduce = sum(permute_1, dim=[0, 1])
    out = relu(add) * permute(permute_1, indices=(2, 1, 0))
    return reduce, out

tensors = {
    "accum": Tensor(torch.float32, shape=(128, 256)),
    "bias":  Tensor(torch.float16, shape=(4, 64)),
    "reduce": Tensor(torch.float32, shape=(128,)),
    "out":   Tensor(torch.float16, shape=(128, 4, 64)),
}
epilogue = python_ast.trace(example_epilogue, tensors)
```

**Shape/Type 传播（Shape Type Propagation）。** 这个 pass 分两个阶段。第一阶段按拓扑序
访问节点，填充缺失的 stride、shape 与 dtype。第二阶段按逆拓扑序更新 shape 与 stride 以
消除隐式广播。例如 `bias` 的 shape 与 stride 从 `(4,64):(64,1)` 更新为
`(128,4,64):(0,64,1)`，第一维被广播。

**Layout 节点消除（Layout Node Elimination）。** 这个 pass 通过移除 layout 节点来规范化
IR。在 epilogue visitor 抽象下，所有算子必须与生产者的空间循环对齐，在我们的 IR 中这
等价于所有节点与 `accum` 节点具有相同 shape。然而如图 11(B) 所示，layout 节点会改变
shape，违反这一要求。我们通过让 layout 节点与邻居交换、把它们从 `accum` 处「推」开来
消除它们；这些 layout 节点最终会通过更新 shape 与 stride 合并进某个 load 或 store 节点。
结果见图 11(D)：所有算子现在共享同一 shape。

**获取实现与 Visitor（Get Implementation and Visitor）。** 这个 pass 把规范 IR lower 到
具体实现。虽然为简单起见我们的 IR 只有四类节点，每一类都可以有针对不同 stride 的多个底层
实现。例如 `reduce` 的 stride 是 `(1,0)`，把矩阵归约成列向量；而 `out` 的 stride 是
`(256,1)`，按 row-major 把矩阵存回内存。在这个 pass 中，我们根据每个节点的 stride 推断
其实现，并根据出度为各非叶节点注入 visitor。结果见图 11(D)。

**C++ Emitter。** 我们的编译器提供一个包含常用 epilogue 算子的高性能算子库。C++ emitter
只需遍历图 11(D) 的 IR 并按后序 emit 每个节点，就能 emit epilogue 的模板类。值得注意的
是，对于我们库未覆盖的算子，在 epilogue visitor 抽象下，它们可以被轻松实现或由编译器
技术生成。

### 5.3 附加特性

本节总结 EVT 提供的、用于提升灵活性与性能的附加特性。

**动态 shape。** 我们的 epilogue 算子把 shape 与 stride 作为运行时参数，因此同一个编译
出的 kernel 可以支持动态 shape 的输入。

**性能。** EVT 与 StreamK 兼容以获得更好的负载均衡。它还在 Ampere GPU 上为 epilogue
引入 ping-pong buffer，使 epilogue 中的计算与内存访问得以重叠。

> 译注：这对应 CUTLASS 里 `EpilogueWithVisitorCallbacks` 的 `Stages` 模板参数
> （`Stages <= 2`，即两级 smem ping-pong）以及 `DefaultGemmWithVisitor` 对
> `ThreadblockSwizzleStreamK` 的 `SelectBase` 特化。

**主循环融合。** 虽然 EVT 聚焦 epilogue 融合，它也为特定生产者（例如矩阵乘）提供融合
主循环参数 permutation 的能力。具体来说，CUTLASS 用运行时参数提供的 stride 来索引矩阵乘
的被乘数与乘数；我们利用这一特性重新计算 permute 后输入的 stride，从而在不真正 permute
内存中数据的前提下保证被乘数与乘数被正确访问。

## 6 评估

本节在五个真实 NN 架构上评估 EVT 并与现有编译器比较。6.2 节评估端到端训练性能；6.4 节
给出各架构代表性层的额外评估以解释 EVT 加速的来由，同时也使得与 TVM 这类推理编译器的
比较成为可能。

### 6.1 实验设置

**Benchmark。** 选取 BERT-Large、VIT、ResNet-50、XML-CNN、GCN 以覆盖多样架构与领域。

- **BERT-large**（语言建模）与 **VIT**（计算机视觉）由堆叠的 self-attention 层构成，已成
  现代神经网络不可或缺的构件。这两个 benchmark 覆盖 GEMM、Batched GEMM、Softmax、
  LayerNorm 等多种主循环，其 epilogue 由各种逐元素运算、激活函数、归约、reshape 与
  permutation 组成。
- **ResNet-50** 是 CV 任务最流行的骨干网之一。除 GEMM 外它还包含 `Conv2dFprop`、
  `Conv2dDgrad` 等特殊重运算；此外由于 BN 运算的复杂性，其 "Conv-BN-ReLU" 模式难以融合。
- **XML-CNN** 是极端分类任务的代表，标签空间含数百万类别，其性能对语言建模与推荐系统等
  应用至关重要。由于类别数极多，XML-CNN 在优化损失函数（二元交叉熵）、其梯度 kernel
  以及周边逐元素函数与归约上提出了新挑战。
- **GCN**（节点分类）作为建模图结构数据的图神经网络（GNN）代表被选中。GCN 在其聚合阶段
  引入新的主循环运算 SpMM；此外全图训练的大 batch size 会给 U-turn 优化带来挑战。

**平台。** 单张 NVIDIA A100 GPU（40 GB），CUDA 12.1，NGC docker
`nvcr.io/nvidia/pytorch:23.07-py3`。

**Baseline。** 端到端 benchmark 中，EVT 与 Torch Inductor 和 NVFuser 比较。前者集成了从
Triton 到 CUTLASS 的多种 SOTA 编译器技术，我们把 mode 设为 `max-autotune` 以取得最优
性能。NVFuser 是一个 GPU 算子编译器，JIT 编译快速灵活的 GPU 专用代码。逐层 benchmark 中
我们与 Triton 和 TVM 比较；对 TVM，GEMM 与卷积用 autotvm 调优以利用 Tensor Core，其余
通过 Ansor 生成。为公平比较，所有方法都应用 CUDA Graph 与 AMP。

### 6.2 端到端 Benchmark

五个 benchmark 的端到端训练加速汇总在图 12，结果对朴素 PyTorch 实现归一化。我们的 EVT
在所有 benchmark 上取得 1.23–3.10× 的端到端训练加速，优于现有编译器。

### 6.3 消融实验

为展示各优化的贡献，图 12 也包含了消融实验结果。

**消融设置。** 「损失消除」「分解」「归约消除」的消融通过从 pass manager 中排除相应 pass
实现。「ILP 划分器」方面，使用一个替代的朴素划分器：沿用已有研究做法，先把计算图切成
前向与反向，再用受 TVM 启发的简单启发式找可融合子图。「EVT 算子编译器」的消融把图级优化
接进 torch inductor 后端，由 inductor 完成算子级优化。注意 GCN 被排除，因为 inductor
不支持稀疏计算。

**损失消除的贡献。** 损失消除 pass 对那些损失函数及其反向占执行时间比重较大的模型影响
显著。例如在 XML-CNN 中它移除了二元交叉熵损失；在 GCN 中它使图 2 描述的 softmax 与交叉
熵损失的优化成为可能。这表明损失消除是一种有效的损失函数优化策略，而先前研究中没有这一
项。

**分解的贡献。** 分解为融合创造了额外机会。以 ResNet-50 为例，分解 batch normalization
带来了额外融合机会，产生图 12 所示的显著性能提升。

**归约消除的贡献。** 在 GCN 中，归约消除通过使 softmax 与分解后的 `nllLoss` 反向之间的
融合成为可能而提升性能，如图 2 所示。

**划分器的贡献。** 图 12 中不使用 ILP 划分器的结果在全部五个模型上都劣于 EVT。值得注意
的是，除性能考虑外，基于 ILP 的形式化还提供了集成额外目标函数、约束与启发式的灵活性与
可扩展性。

**EVT 算子编译器的贡献。** 图 12 表明，即便纳入了 EVT 的图级优化，inductor 的性能仍劣于
EVT。这说明 EVT 的算子编译器生成的融合 kernel 性能更优。

### 6.4 逐层 Benchmark

本节在五个模型的代表性层上评估 EVT。对每个 benchmark，我们可视化 Triton、TVM 与 EVT 的
计算图；被融合的算子用框圈起，橙色表示重算子（如 GEMM、Softmax），绿色表示轻算子
（逐元素 / 归约）。我们也给出对 PyTorch 归一化的延迟，并拆成 heavy 与 light ops 两部分，
同时标注加速比。

**Self-Attention 层（BERT、VIT）。** 加速 self-attention 层要求编译器能够 1) 高效处理
同一张量的不同 permutation 与 reshape；2) 生成高性能的融合 kernel。EVT 通过 5.2 节基于
cute 的 IR 设计与相应 pass 解决 1)，通过 Epilogue Visitor 抽象保证 2)。相比之下，Triton
生成的融合 kernel 效率不如我们，而 TVM 未能高效处理 1)。（图 13：EVT 1.54×，Triton
1.16×，TVM 0.85×。）

**MLP（BERT、VIT）。** 优化 MLP 要求编译器找到计算图的可行且最优划分。图 14 中，EVT 通过
第 4 节的划分器识别出 ReLU 反向分解出的两个节点之间的最优切分，而 Triton 与 TVM 产生
次优划分。（图 14：EVT 1.16×，Triton 1.07×，TVM 0.70×。）

**Conv-BN-ReLU 层（ResNet）。** channel-last 布局下 BN 的主要挑战是对归约维的 strided
访问，所以优化该层要求编译器 1) 分解 batch norm 算子以创造新的融合机会；2) 在 channel-last
布局下生成高性能 kernel。EVT 把 BN 分解成归约与逐元素运算，并把归约高效地融合到前面的
卷积中。相比之下，Triton 融合的 BN-ReLU 因布局问题性能较差，TVM 未能融合归约。
（图 15：EVT 1.24×，Triton 1.11×，TVM 0.61×。）

**二元交叉熵损失（XML-CNN）。** 加速 BCE 损失涉及 1) 在前向与反向的边界上识别融合机会；
2) 生成带多样 epilogue 算子（广播、各种逐元素算子、归约）的高性能融合 GEMM。EVT 通过
损失消除 pass 完成 1)，通过 Epilogue Visitor 抽象达成 2)。相比之下，Triton 未能跨越前向
与反向图的边界，TVM 生成的 kernel 性能较差。（图 16：EVT 4.13×，TVM 3.08×，Triton
2.73×。）

**Softmax 交叉熵损失（GCN）。** 这个图与 BCE 损失有类似的优化需求。凭借图级优化与算子
编译器，EVT 把整个图融合成单个 kernel，相对 cuDNN 取得 2.81× 加速。相反，Triton 未能
识别前向与反向图之间的融合机会，TVM 生成的 kernel 效率较低。（图 17：EVT 2.81×，
TVM 0.92×，Triton 0.85×。）

## 7 结论与讨论

EVT 使得在容纳多样融合模式的同时生成 SOTA 性能的融合 kernel 成为可能。配合充分释放其
潜力的图级优化与划分器，EVT 自动优化深度学习训练负载并达到 SOTA 性能。

**对多 GPU / 多节点训练的影响。** 尽管本文主要关注单 GPU，EVT 也可以轻松用于加速多 GPU
/ 多节点训练。特别是最近的一个趋势是把 GEMM 与后继的通信 kernel 融合，从而在 tile 粒度
上重叠计算与通信。这一技术原本需要专家手工实现融合 kernel；有了 EVT，通信算子可以实现为
一类特殊的 store 节点，同时复用 EVT 的其余基础设施。

## A 制品附录（摘要）

制品包含 EVT 源码，含搭建环境的 Dockerfile 与复现论文图 12–17 加速比的 bash 脚本。

- 程序：`EVT_AE/python` 是 EVT 编译器源码，`EVT_AE/src` 是额外的 CUDA 源码，
  `EVT_AE/benchmark` 是评估用的 benchmark；
- 数据集：ogbn-mag（其余用合成输入）；
- 运行环境：Dockerfile 提供全部依赖，结果在 driver 530.30.02、CUDA 12.1 下评估；
- 硬件：需要单张 A100-SXM4-40GB；
- 实验：`figure12.sh` 与 `figure13_17.sh`；
- 磁盘：< 50 GB；构建镜像 < 20 分钟；跑完实验约 2 小时；
- 公开获取：`github.com/apuaaChen/EVT_AE`、`zenodo.org/doi/10.5281/zenodo.10790585`。

## 译文遗留问题

- 4.1 节公式 (3a)–(3j) 的精确形式在文本抽取中丢失下标，译文只说明各式作用。要复现 ILP
  形式化需回到 PDF p.306。
- 算法 1、算法 2 的伪代码在抽取中错行严重，译文按正文描述重述了它们的步骤，未逐行照译。
- 图 12–17 的柱状图数值由抽取文本中的标注读出（如 self-attention 1.54×），图中未标注的
  细分项（heavy / light ops 拆分）无法从文本还原。

## Profile

<!-- 译文篇，无实测。 -->

## 后记

<!-- HUMAN: 读完补充自己的结论、踩坑和下一步。 -->
