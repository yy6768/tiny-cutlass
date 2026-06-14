# FlashAttention 论文导读

这个目录用来放 FlashAttention 1 到 FlashAttention 3 的原始论文 PDF 和阅读笔记。这里不是实现文档，而是后续写 CUTLASS/CuTe kernel 时的背景索引：我们要从论文里抽出算法动机、硬件假设、可验证的实现约束，以及可以转成实验 variant 的优化方向。

## 本地文件

| 论文 | 本地 PDF | 官方来源 |
| --- | --- | --- |
| FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | `flashattention-1-2205.14135.pdf` | https://arxiv.org/abs/2205.14135 |
| FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | `flashattention-2-2307.08691.pdf` | https://tridao.me/publications/flash2/flash2.pdf |
| FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision | `flashattention-3-2407.08608.pdf` | https://tridao.me/publications/flash3/flash3.pdf |

Dao-AILab 官方仓库目前把 FlashAttention 和 FlashAttention-2 作为主线实现；FlashAttention-3 在 README 里仍标成 Hopper beta release，面向 H100/H800，已经释放 FP16/BF16 forward/backward 和 FP8 forward，要求 CUDA >= 12.3，并推荐 CUDA 12.8。这个状态很重要：如果我们在 Ada/Ampere 上做 tiny-cutlass 学习，FA3 不是直接可复刻目标，而是 SM90 路线的设计参考。

## FlashAttention 1

### 摘要

FA1 的核心主张是 exact attention，不靠近似稀疏或低秩，而是改变计算顺序：把 `QK^T -> softmax -> PV` 拆成 SRAM tile 上的 streaming 计算，避免把完整的 `N x N` score/probability 矩阵写回 HBM。这里第一次把 attention 的瓶颈明确转成 IO 问题。

### 1 引言

引言把问题摆得很清楚：标准 attention 的计算量是二次的，内存访问也是二次的；当 sequence length 变长时，HBM 读写中间矩阵比单纯 FLOPs 更难受。FA1 的价值不是“少算很多 FLOPs”，而是减少 HBM traffic，让 GPU 在更接近 GEMM 的路径上工作。

### 2 背景

这一节把算法放回 GPU memory hierarchy 里看。我们后续写 kernel 时也应该保持这个视角：global memory、shared memory、register、Tensor Core/普通 CUDA core 之间的流量比单个公式更重要。

### 2.1 硬件性能

这一节说明 GPU 的算力和带宽并不对称，HBM 访问昂贵，SRAM/shared memory 便宜得多。对 CUTLASS 实验来说，这直接对应 tile shape、shared memory residency、register pressure、occupancy 的权衡。

### 2.2 标准 attention 实现

标准实现会 materialize `S = QK^T` 和 `P = softmax(S)`。这非常适合作为 `00` baseline：它语义直观，容易和 PyTorch 对齐，但会暴露 HBM 中间矩阵的问题。

### 3 FlashAttention 算法

算法把 K/V 分块流过来，每个 Q block 维护 row-wise running max 和 running sum。每来一个 K/V tile，就更新当前 block 的 softmax 归一化状态和输出累积。这里的关键不是“用了 online softmax”这句话，而是 online softmax 让我们可以不保存完整 `P`。

### 3.1 算法

实现上要维护每行的 `m_i`、`l_i` 和输出累积 `O_i`。当新的 tile 得到局部 score 后，先更新新的 row max，再用 `exp(m_old - m_new)` 重新缩放旧的分母和旧的输出。这个公式是我们审查 `01/02` 时最应该盯住的地方：任何少一次 rescale，都可能在数值上悄悄错。

### 3.2 IO 复杂度分析

这一节给出 FA1 的理论核心：标准 attention 对中间矩阵的 HBM 访问是 `O(N^2)`，FlashAttention 通过 tiling 把主要中间状态留在 SRAM/register 中，降低 IO complexity。后续 profiling 不能只看 TFLOPS，还要看 DRAM throughput、L2 hit、shared memory traffic 是否符合这个方向。

### 3.3 扩展：Block-Sparse FlashAttention

FA1 也讨论 block-sparse 扩展。对我们当前 tiny-cutlass 来说，短期不必马上做 sparse，但可以保留一个抽象接口：mask/score-modification 最好能在 tile 级别表达，而不是先 materialize 一个大 mask。

### 4 实验

实验不是证明“任何 attention 都快”，而是说明长序列和 memory-bound 场景收益明显。我们自己做 benchmark 时也应该分组：small N、large N、D=64/128/256、causal/non-causal，不能只拿一个 case 下结论。

### 4.1 用 FlashAttention 加速模型

这一节强调模型级速度。对 tiny-cutlass 现在更现实的做法是先证明单 kernel reference parity，再把 kernel 放进一个完整 MHA/SDPA path 里看端到端收益。

### 4.2 更长序列带来的模型收益

长上下文的价值来自 memory savings。我们做 00/01/02 时，可以用 N 增长曲线检查中间矩阵 materialization 是否已经成为主瓶颈。

### 4.3 Attention benchmark

benchmark attention 不能只看算子时间，还要看 batch、head、sequence、head_dim、dropout、causal mask。这里应该和 PyTorch SDPA/official flash-attn 对齐输入形状。

### 5 局限和未来方向

FA1 的局限包括 kernel 特化、硬件适配、backward 和更多 attention 变体。对我们的路线来说，FA1 更像“IO-aware attention 的最低正确闭环”，不是最终性能答案。

## FlashAttention 2

### 摘要

FA2 的重点从“减少 HBM IO”推进到“更好地使用 GPU 并行性”。它仍然保留 FA1 的 IO-aware 思路，但把性能问题拆到 parallelism、warp partitioning、non-matmul FLOPs 上。

### 1 引言

FA2 指出 FA1 仍没有充分利用 GPU，尤其是当 batch/head 数量不足、sequence 或 head_dim 组合不理想时。这个观点和我们看到的低 residency/low issue active 很相关：不是融合以后自然就快，还要让并行粒度足够细。

### 2 背景

背景重新梳理 attention、GPU 硬件和 FA1 的算法结构。对实现者来说，这一节提示我们不要把 softmax 当成孤立 kernel 优化；它必须和两个 GEMM tile 的数据路径一起看。

### 2.1 硬件特性

FA2 更强调 Tensor Core、shared memory、register、warp 之间的分工。对应 CUTLASS/CuTe，就是要从 threadblock tile、warp tile、MMA atom、copy atom、pipeline stage 这些层次描述实验。

### 2.2 标准 attention 实现

标准 attention 仍是对照组。我们自己的 `00` 可以作为语义 baseline，但性能路线应该快速走向 tiled/online/fused，否则只是在测中间矩阵写回的代价。

### 2.3.1 前向

forward 重新组织了 Q/K/V tile 的遍历方式，并减少不必要的 non-matmul work。这里可以映射到我们的 forward-only 实验：先把 online softmax 和 `P @ V` 的累积公式做对，再讨论 tile 和 warp 的拆法。

### 2.3.2 反向

backward 的思路是 recompute attention probability，而不是存下大矩阵。tiny-cutlass 当前如果先做 forward，可以先不展开 backward；但文档和接口最好不要把中间 `P` 当成未来 backward 必需品。

### 3.1 算法

FA2 算法层面减少 rescale、bound check、mask 等非矩阵乘开销，同时改善 Q-block 并行。这里对我们有两个推进方向：先证明每个 tile 的 online recurrence 正确，再把多 CTA/多 warp 的 work partition 做成可比 variant。

### 3.1.1 前向

forward 的实作重点是如何让 `QK^T` 和 `PV` 都尽量落到 Tensor Core 路径，同时把 softmax 的 scalar work 控制住。Nsight 上要同时看 tensor pipe、issue active、dram/l2，而不是单看一个“utilization”。

### 3.1.2 反向

FA2 backward 的核心仍是 IO tradeoff：多算一点，少存很多。未来如果我们做 backward，reference parity 应该包括 dQ/dK/dV，并且误差阈值要按 dtype 和 accumulation 策略单独设。

### 3.2 并行性

这一节是 FA2 的主菜之一：当 batch/head 不够多时，只按 batch/head 分块会不够，需要沿 sequence 维度增加并行。LeetCUDA 里的 `split-q` 思路就和这个方向接近。

### 3.3 warp 之间的 work partitioning

FA2 讨论 warp 内/warp 间如何分配工作，减少通信和 shared memory 往返。我们复刻时不要只说“用 4 warps”，要写清楚每个 warp 持有 Q 的哪几行、KV 的哪段、P/V 累积如何交换。

### 4 实验验证

这一节提醒我们 benchmark 必须和 reference 对齐。tiny-cutlass 的通用规则应该是 `build -> verify -> bench -> profile`，verify 没过就不谈性能。

### 4.1 Attention benchmark

FA2 的 benchmark 覆盖不同 sequence/head_dim/causal 组合。我们的路线图应该保留 matrix：`N in {512, 2K, 8K, 16K}`，`D in {64,128,256,512}`，并明确哪些 shape 是 FA2 支持范围，哪些是 SDPA/math/cuDNN fallback。

### 4.2 端到端性能

端到端性能会受到 QKV projection、output projection、dropout、mask、KV cache、framework dispatch 影响。单 kernel 快并不自动等于模型快，所以最终要接 PyTorch extension 或 C++ harness 做端到端测量。

### 5 讨论和未来方向

FA2 的未来方向包括更多硬件、更多特性和进一步减少非 matmul 开销。对我们来说，FA2 是 SM80/SM89 上最重要的对照线。

## FlashAttention 3

### 摘要

FA3 把焦点放到 Hopper/SM90：WGMMA、TMA、warp specialization、FP8。它不是 FA2 的小改，而是硬件能力变了以后重新设计 pipeline。

### 1 引言

引言指出 FA2 在 H100 上没有吃满理论峰值，原因之一是新硬件特性没有充分利用。这个对 tiny-cutlass 的启发是：SM80/89 和 SM90 应该分路线，不要强行把同一个 kernel 通过 `#if` 拼成万能实现。

### 2 背景：Multi-Head Attention 和 GPU 特性

这一节把 MHA 和 Hopper execution model 放到一起。SM90 的核心关键词是 warpgroup 级矩阵乘、异步 global-to-shared 搬运、producer/consumer 分工。

### 2.1 Multi-Head Attention

数学仍是 `softmax(QK^T / sqrt(d))V`，但 FA3 关心的是如何把这件事拆成 Hopper 能并行推进的 pipeline。也就是说，公式没变，调度模型变了。

### 2.2 GPU 硬件特性和执行模型

Hopper 的 WGMMA 是 warpgroup 级别，TMA 能降低地址计算和搬运开销，FP8 Tensor Core throughput 更高。写 CuTe kernel 时，这会落到 `TiledMMA`、`TMA copy`、barrier、pipeline stage 的设计上。

### 2.3 标准 attention 和 FlashAttention

这里回顾 FA1/FA2 仍然为 IO-aware baseline 服务。FA3 不是否定前两代，而是在“不 materialize P”的基础上继续处理异步和低精度。

### 3 FlashAttention-3 算法

FA3 的算法核心是 pipeline co-design：让 GEMM、softmax、TMA 尽可能重叠，同时让 FP8 的量化误差可控。它要求我们把 Nsight Systems 和 Nsight Compute 一起看：前者看时间线/overlap，后者看每个 kernel 的 pipe utilization。

### 3.1 通过 warp-specialization 和 pingpong scheduling 实现 producer-consumer 异步

producer warpgroup 负责 TMA/load，consumer warpgroup 负责 WGMMA/compute。pingpong scheduling 让不同 warpgroup 的 GEMM 和 softmax 交错。这个方向更适合作为 SM90 专门 variant，而不是强行塞进 SM80 实现。

### 3.2 warpgroup 内 GEMM 与 softmax 重叠

FA3 进一步在 warpgroup 内 overlap GEMM 和 softmax。收益来自隐藏 exp/softmax 的特殊函数开销，但代价是 register pressure 上升。这里必须用 Nsight 验证：如果 register 爆了导致 occupancy 太低，理论 overlap 可能会反咬一口。

### 3.3 FP8 低精度路径

FP8 提供更高 Tensor Core throughput，但 activation outlier 会放大量化误差。FA3 使用 incoherent processing/Hadamard transform 把 outlier 打散，并指出这个变换可以和 rotary 这类 memory-bound 操作融合。这个思路会直接进入我们的量化融合设计文档。

### 4 实验验证

FA3 的实验对照包括 FA2、Triton、cuDNN。我们自己的实验如果要引用 FA3 风格，至少要给出 `.ncu-rep`、`.nsys-rep` 和 CSV，不能只贴一个 time。

### 4.1 Attention benchmark

FA3 把 FP16/BF16 和 FP8 分开看。tiny-cutlass 后续也应该分 dtype report：FP16/BF16 correctness、FP8 correctness、FP8 with transform/scaling correctness，不混在一张表里。

### 4.2 消融：2-stage pipeline 实验

ablation 的价值是证明每个 pipeline 决策真的有收益。我们设计 `00/01/02/...` variant 时也应该这么做：每一版只引入一个主要变量，profiling 能解释它为什么变快或变慢。

### 4.3 数值误差验证

低精度 attention 最终不能只看速度。FP8 路线要单独记录 MAE/max error、scale granularity、outlier 处理、reference dtype，以及是否对输出分布造成系统偏差。

### 5 讨论、局限和结论

FA3 的结论是硬件感知算法设计会继续变重要。对 tiny-cutlass 来说，这意味着文档、kernel、profile 都应该按架构分层：SM80/89 先打牢 FA2/CUTLASS 2.x/3.x 思路，SM90 再单独研究 TMA/WGMMA/FP8 pipeline。

## 对 tiny-cutlass 的直接结论

1. `00` 应该保留为标准 attention baseline，用于解释 materialized `S/P` 的代价。
2. `01/02` 应该围绕 online softmax、tiled IO、`P` 不落 HBM 的路径推进，先 correctness，再 benchmark。
3. SM80/SM89 重点学习 FA2：parallelism、warp partitioning、shared memory layout、cp.async pipeline。
4. SM90 重点学习 FA3：CuTe/TMA/WGMMA、warp specialization、GEMM/softmax overlap、FP8。
5. 所有性能结论必须绑定 reference parity 和 Nsight artifact；没有 `.ncu-rep`/`.nsys-rep`/CSV 的数字只能作为临时观察。

## 资料来源

- Dao-AILab official repository README: https://github.com/Dao-AILab/flash-attention
- FlashAttention paper: https://arxiv.org/abs/2205.14135
- FlashAttention-2 paper: https://tridao.me/publications/flash2/flash2.pdf
- FlashAttention-3 paper: https://tridao.me/publications/flash3/flash3.pdf
- FlashAttention-3 official blog: https://tridao.me/blog/2024/flash3/
