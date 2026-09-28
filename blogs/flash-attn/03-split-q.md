# CUTLASS 学习记：03 Split Q

## 前言

这篇接着 02-split-kv，参考 FlashAttention-2 原文和 LeetCUDA 的 `flash_attn_mma_split_q.cu`，用 CUTLASS 2.x 标准组件实现 split-Q。重点看三件事：减少非 matmul 计算、沿 seqlen 增加并行，以及改变 warp 分工来避免 split-K 的通信开销。

实现继续按 example 13 的 `B2bMmaBase` / `B2bMmaMultistage` 划分职责。输入保持和 02 相同，目标是 MAE < 1e-6、耗时小于 02 的 0.95 倍；是否达到，以本次验证和实测为准。

## Overview

1. **从论文 Algorithm 1/2 看非 matmul 开销。** 推出保留未归一化输出、最后一次归一化和只保存 LSE 的递推，再读 backward 怎样用 LSE 重算 P。
2. **CTA 沿 seqlen 并行。** 对照 02 的“外 K/V、内 Q”与 03 的“外 Q、内 K/V”；说明 backward 固定 KV 块后 dQ 为什么需要跨 CTA 汇总。
3. **Warp 沿 Q 分工。** 先看论文中 split-K 与 split-Q 的区别，再核对当前 02 与 03 的 warp 实际所有权，避免把已有的性质算成新收益。
4. **沿 CUTLASS 调用链读实现。** device policy → kernel 初始 iterator → threadblock 主循环 → 标准 warp MMA → epilogue。标出自定义 online softmax，核对 example 13 的对应位置。
5. **精度与耗时如何验收。** 同输入、同 cuDNN reference、严格 MAE；候选记录、失败原因，以及四个已有 benchmark shape 的成对计时。

这里给出阅读顺序。当前 02 已有按 Q 行分工的 warp，但仍是每个 batch/head 一个 CTA，外层扫描 KV、内层扫描全部 Q，并且每个 KV 轮次都归一化 O；03 对应的变化要以实际源码为准。

阅读入口：[FlashAttention-2 原文](https://arxiv.org/pdf/2307.08691) §3.1–3.3、Algorithm 1/2；[LeetCUDA split-Q](https://github.com/xlite-dev/LeetCUDA/blob/main/kernels/flash-attn/mma/basic/flash_attn_mma_split_q.cu)；CUTLASS example 13 的 [B2bMmaBase](../../3rdparty/cutlass/examples/13_two_tensor_op_fusion/threadblock/b2b_mma_base.h)、[B2bMmaMultistage](../../3rdparty/cutlass/examples/13_two_tensor_op_fusion/threadblock/b2b_mma_multistage.h) 和 [DefaultB2bMma](../../3rdparty/cutlass/examples/13_two_tensor_op_fusion/threadblock/default_b2b_mma.h)。

下面先读论文和源码，最后才看测量；不以 Profile 的指标倒推算法。本文项目内的 S 包含 softmax scale，论文正文为简洁暂时省略这个系数。实现只覆盖 FP16、D=Dv∈{32,64,96,128}、非 causal、无 dropout 的情况；论文讨论的 causal、MQA/GQA 不在本次接口内。

## 1. Algorithm 1/2：先把非 matmul 工作理清

论文 §3.1 给的动机是硬件吞吐差异：它以 A100 为例，FP16/BF16 Tensor Core matmul 的理论峰值是 312 TFLOPs/s，普通 FP32 运算是 19.5 TFLOPs/s。16 倍是这两个**理论峰值**的比值，不是本机 RTX 4070 的实测加速。QK 和 PV 两次矩阵乘法仍然要做，所以这里先看怎样少做夹在两次 MMA 中间的逐点工作。[论文 §3.1](https://arxiv.org/pdf/2307.08691)

我先固定一行 Q 看在线 softmax。设这一行访问到第 j 块 KV 前，最大值、指数和、未归一化输出分别为 m、l、u。新块的 score 是 S_j = scale·Q_i K_j^T。先求 m' = max(m, rowmax(S_j))，再令 α = exp(m-m')、P̃_j = exp(S_j-m')。更新式是：

~~~text
l' = α·l + rowsum(P̃_j)
u' = α·u + P̃_j V_j
最终 O_i = u/l，L_i = m + log(l)
~~~

关键是 u 在循环中始终不除以 l。上一轮的贡献只需乘 α，本轮 P̃_j V_j 加进去；除法等到整个 KV 序列结束才做。论文 §3.1.1 / Algorithm 1 正是用这种写法减少逐 tile 的逐点缩放，并只保存一份行级 LSE = m+log(l) 供 backward 使用。[论文 §3.1.1、Algorithm 1](https://arxiv.org/pdf/2307.08691)

这与当前 02 的差别能从源码直接看见：[02 threadblock](../../csrc/flash-attention/02-split-kv/threadblock/flash_attn_mma.h) 每个 KV 轮次生成 tile 的最大值与行和，计算本轮 P̃V 后，将旧 O 和新贡献合并并除以新的 l；中间 O、m、l 写进 FP32 workspace，下一轮再取回。[03 threadblock](../../csrc/flash-attention/03-split-q/threadblock/flash_attn_mma.h) 把输出 accumulator、m 和 l 留在负责这块 Q 的 CTA 内；[03 epilogue](../../csrc/flash-attention/03-split-q/epilogue/flash_attn_epilogue.h) 才做最终除法。B08 还把行和保存在各 lane，扫描完 KV 后才做一次 warp 内求和。这是论文目标在本实现中的具体落点；不能把省掉的每次归一化等同于省掉所有 softmax 工作。

Backward 为什么只需要 L？若前向保存 L_i = log Σ_k exp(S_ik)，便能在访问任一 KV 块时重算 P_ij = exp(S_ij-L_i)，不必在 HBM 留下整张 P。设上游梯度 dO_i，D_i = rowsum(O_i ⊙ dO_i)，dP_ij = dO_i V_j^T，那么 softmax 的梯度是 G_ij = P_ij ⊙ (dP_ij-D_i)。因为本文把 scale 放进 S，回传 QK^T 时还要乘 scale：

~~~text
dV_j += P_ij^T dO_i
dK_j += (scale·G_ij)^T Q_i
dQ_i += (scale·G_ij) K_j
~~~

论文 Algorithm 2 给出 D、重算 P、dV/dK/dQ 的顺序。[论文 §3.1.2、Algorithm 2](https://arxiv.org/pdf/2307.08691) 当前实现的 [Delta 预处理](../../csrc/flash-attention/03-split-q/kernel/flash_attn_backward_prepare.h) 求 D；[backward threadblock](../../csrc/flash-attention/03-split-q/threadblock/flash_attn_backward_mma.h) 先以 KQ^T 和 VdO^T 得到 score/dP，再重算 P、计算 dS，最后用三个 TensorOp 分别形成 dV、dK、dQ。这里的 online softmax、D 和 dS 是 attention 自定义逻辑，不是 CUTLASS 自带的 GEMM epilogue。

## 2. 从一个 head 到多个 CTA：交换 Q/KV 循环

论文 §3.2 的动机很具体：长序列、小 batch 时，如果只有 B×H 个 CTA，就算每个 CTA 很忙，也难以让更多 SM 同时工作。前向可以按 Q 行分块；每个 CTA 独占一个 Q_i 和 O_i，内部遍历所有 K_j/V_j，CTA 之间无需合并输出。[论文 §3.2、Figure 2](https://arxiv.org/pdf/2307.08691)

当前 02 的 [kernel](../../csrc/flash-attention/02-split-kv/kernel/flash_attn.h) 使用 grid=(batch, head)，[threadblock](../../csrc/flash-attention/02-split-kv/threadblock/flash_attn_mma.h) 则是外层 KV、内层所有 Q：K/V 一次装入 shared 后供多个 Q tile 使用，但 Q 和行状态会在下一块 KV 时重新读写。03 的 [kernel](../../csrc/flash-attention/03-split-q/kernel/flash_attn.h) 改为 grid=(ceil(Sq/64), head, batch)；每个 CTA 装自己的 Q tile 一次，内层轮流搬入 K/V，完整扫描 Sk 后写 O。比如 B=1、H=4、Sq=1024 时，CTA 数从 4 变成 64。代价是不同 Q CTA 会重复读取 K/V；是否更快仍由实测判定。

Backward 倒过来分工：固定一块 KV 给一个 CTA，内层遍历所有 Q。于是 dK_j、dV_j 只有这个 CTA 写；不同 KV CTA 都会给同一个 dQ_i 加贡献。[论文 §3.2、Algorithm 2](https://arxiv.org/pdf/2307.08691) [03 backward kernel](../../csrc/flash-attention/03-split-q/kernel/flash_attn_backward.h) 对应 grid=(ceil(Sk/64), head, batch)。实现先清零 FP32 dQ workspace，CTA 用 [atomic epilogue](../../csrc/flash-attention/03-split-q/epilogue/flash_attn_backward_epilogue.h) 汇总各 KV 块贡献，最后再转 FP16。这里的 atomic 是 CTA 之间对 dQ 的合并；它不意味着一个 warp 把自己的 MMA 归约维拆给其他 warp。

## 3. Warp 如何分工：论文的 split-Q 与当前代码

论文 §3.3 / Figure 3 对比的起点是早期 FlashAttention：多个 warp 各处理 K/V 的一片，却要把同一批 Q 行的 O 部分和相加，因此需要把部分结果放进 shared、同步，再跨 warp 合并。FA2 改为按 Q 行分 warp；K/V 供所有 warp 读取，每个 warp 从 QK、softmax 到 PV 都负责自己那片输出行，不需要跨 warp 汇总 O。[论文 §3.3、Figure 3](https://arxiv.org/pdf/2307.08691) 指定的 [LeetCUDA split-Q](https://github.com/xlite-dev/LeetCUDA/blob/main/kernels/flash-attn/mma/basic/flash_attn_mma_split_q.cu) 也按 warp 的 Q/P 行坐标组织 MMA；本文只取分工思路，不把其手写 HMMA 和输入假设搬成项目 API。

这里容易把“论文相对早期 FA 的改动”和“03 相对本仓库 02 的改动”混在一起。读过当前 [02 policy](../../csrc/flash-attention/02-split-kv/kernel/default_flash_attn.h) 与 [02 threadblock](../../csrc/flash-attention/02-split-kv/threadblock/flash_attn_mma.h) 后，结论是：02 的 warp 已经拿互不重叠的 Q 行，而且单个 warp 的 QK 覆盖完整逻辑 KV tile、PV 覆盖完整输出通道，不做跨 warp 的 O 归约。03 沿用这个性质；它本轮真正改变的是 CTA 外内循环顺序、行状态的驻留与最终归一化，以及 V 的 shared layout。不能把论文 Figure 3 的全部收益重新算在 02→03 上。

当前 03 的 [forward policy](../../csrc/flash-attention/03-split-q/kernel/default_flash_attn.h) 固定 4 个 warp：每 warp 16 行 Q，QK tile 的 N=64 包含完整 KV 列，PV tile 的 N 覆盖逻辑输出宽度。每 warp 的行最大值和行和只需在本 warp 的 lane 间做 shuffle。K/V 搬进同一份 shared，copy 完成和下轮覆写前仍必须做 CTA 同步；“避免 split-K”说的是不用跨 warp 合并部分 O，并不表示主循环没有同步。Backward 有更多输入和梯度依赖，论文也明确说它仍需同步；当前代码每 warp 负责 16 行 KV，dQ 的 2×2 warp 网格按 Q 行与输出通道分工，本 CTA 的完整 64 列 KV 归约由每个相关 warp 自己完成。

## 4. 对照 example 13 读 CUTLASS 调用链

我用 [example 13 的 DefaultB2bMma](../../3rdparty/cutlass/examples/13_two_tensor_op_fusion/threadblock/default_b2b_mma.h)、[B2bMmaBase](../../3rdparty/cutlass/examples/13_two_tensor_op_fusion/threadblock/b2b_mma_base.h) 和 [B2bMmaMultistage](../../3rdparty/cutlass/examples/13_two_tensor_op_fusion/threadblock/b2b_mma_multistage.h) 核对职责：工厂选择标准 iterator/MMA，kernel 放置 CTA 并创建初始 iterator，threadblock 管完整循环、拷贝推进与 shared 生命周期，epilogue 写结果。03 组合这些 CUTLASS 2.x 组件，但没有继承 example 13 的整条双 GEMM pipeline；online softmax 必须自己写。

| 代码位置 | 在这一版负责什么 |
|---|---|
| [device factory](../../csrc/flash-attention/03-split-q/device/flash_attn.cu)、[DefaultFlashAttnSplitQ](../../csrc/flash-attention/03-split-q/kernel/default_flash_attn.h) | 按 arch、D、tile shape 生成 policy；将标准 ThreadMap、global/shared iterator、warp MMA 和 epilogue 接起来。D=96 的 PV 物理通道补到 128，逻辑输出仍为 96。 |
| [forward kernel](../../csrc/flash-attention/03-split-q/kernel/flash_attn.h) | 确定 Q tile/head/batch、KV 循环次数和首个 iterator，调用 Mma 后交给 epilogue；不在这里展开 CTA 主循环。 |
| [forward threadblock Mma](../../csrc/flash-attention/03-split-q/threadblock/flash_attn_mma.h) | Q 在循环前搬一次；每轮用标准 iterator/cp.async 搬 K/V、同步、做 QK、更新 m/l/u、做 PV，再推进 K/V iterator。搬运属于这里的 Mma，不另造 Loader 层。 |
| [forward epilogue](../../csrc/flash-attention/03-split-q/epilogue/flash_attn_epilogue.h) | 用标准 accumulator/output iterator 把 u/l 转成 FP16 O；训练时可另存自然对数 LSE。 |

两次 TensorOp 之间，先在 QK 的 FP32 accumulator fragment 所在寄存器中算 softmax，把值改写为 P̃；随后 [MmaTensorOpFragmentIterator](../../3rdparty/cutlass/include/cutlass/gemm/warp/mma_tensor_op_fragment_iterator.h) 按 PV 的 A fragment 布局读取它，由标准 LinearCombination 做 FP16 概率转换。这是 example 13 从第一次 GEMM accumulator 接到第二次 GEMM 输入的组件用法。QK、PV 的矩阵乘法由 CUTLASS warp MMA 承担；求 max/exp/行和、α 缩放与 mask 属于自定义 attention。V 的 shared layout、ThreadMap 和 iterator 来自标准 DefaultMmaCore；policy 只借用这些组件，每 CTA 保留一块 V tile，不声称实现了三阶段 pipeline。

Backward 沿同一职责划分：[device](../../csrc/flash-attention/03-split-q/device/flash_attn_backward.h) 管 workspace/launch，[kernel](../../csrc/flash-attention/03-split-q/kernel/flash_attn_backward.h) 放置 KV CTA 和初始 iterator，[threadblock](../../csrc/flash-attention/03-split-q/threadblock/flash_attn_backward_mma.h) 完成 Q 循环、五次 TensorOp 与 shared 复用。dQ 的跨 CTA FP32 累加由 [atomic epilogue](../../csrc/flash-attention/03-split-q/epilogue/flash_attn_backward_epilogue.h) 完成。上述路径与论文 Algorithm 2 的数据依赖对得上，但代码的物理 padding、FP16 转换和 workspace 是本项目为 CUTLASS/当前接口做的选择。

## 5. 精度门限先于速度结论

把数学上等价的递推放进 FP16 TensorOp 后，tiling 顺序、P 转 FP16 的时刻、FP32 行和的求和树都会影响末位。本项目的验证仍沿用 02 的 Q/K/V 输入与 cuDNN reference；03 forward 用严格 MAE<1e-6，不能因为公式正确就略过边界与放大输入。C01 因 residue-first 的 KV 顺序在 Sk=65 失败；C02 把完整 tile 放前、尾块放后才通过；C03 改用标准 swizzled V 布局后，四个既有 benchmark shape 的历史测量都低于 0.95×02。具体数值和对应二进制放在后面的 Profile 与[候选记录](../../docs/03-split-q-results.md)，不在算法段落用吞吐率代替证据。

当前源码已迭代到 B08：forward 66 项加 2 项拒绝检查通过，58 项严格输入的最大 MAE 为 4.01022e-7。Backward 已有独立 CTA、五次 TensorOp 和验证 harness，但两边各自运行 training forward、各自保存 O/LSE 后，58 项中仍有 9 项梯度 MAE 超过 1e-6；所以 B08 没有 backward 计时，也不能沿用 C03 的 forward Profile 宣称整个训练路径达标。B07 曾把相同 O/LSE 喂给两边做 58 项诊断，梯度达到门限；这只说明其误差主要落在保存状态的舍入，不等于 B08 通过端到端验收。这个限制会直接影响之后如何判断下一候选，需保留在实验记录里。

## Profile

本节固定记录 C03 forward 的已验证二进制。新增训练状态和 backward 从 B01 开始，后续数值候选记录到 B08；这里的 NCU 和耗时均不覆盖这些改动。B07 共同 O/LSE 的58项梯度诊断达到门限，但它不能替代当前测试采用的独立状态端到端验收。

这轮先遇到的是精度问题。C01 用 CUTLASS iterator 默认的 residue-first 次序，Sk=65 时先算1行、再算64行；第三个用例的 MAE=2.59884e-6，未达到任务要求，因此没有计时。C02 保持算法不变，改成完整块在前、尾块在后，才通过严格验证。FP16 P 的舍入会受到中间最大值和分块次序影响，数学上的等价递推不能直接保证与某个融合 reference 得到相同误差。

最终 C03 在 RTX 4070 Laptop（SM89）上运行；CUDA Toolkit 12.9、driver 592.82、NCU 2025.2。Q/K/V/O 是 FP16 `[B,S,H,D]`，累加与 online state 是 FP32。D=Dv 支持32/64/96/128，无 causal mask 和 dropout。66项验证与2项不支持配置检查通过；其中58项03严格检查最大 MAE=**4.01022e-7**，包含边界、不同seed、Q/K输入缩放及非默认stream上的重复调用。reference始终是原有 cuDNN SDPA，输入生成未改。

历史 C03 验证与计时使用的入口：

```bat
scripts\kernels\attention\attention.bat
```

脚本按 build→verify→bench 执行；当前入口已增加 backward 验证，会在梯度不达标时停止。下面的历史 C03 测量来自新增 backward 之前的成功运行。每次计时先校验本二进制的 parity stamp；固定 seed=3080、input-scale=1，每个样本50次调用，交错运行02/03共5轮，取median。

| (B,H,Sq,Sk,D) | 02 ms | 03 ms | t03/t02 |
|---|---:|---:|---:|
| (1,1,256,256,64) | 0.027439 | 0.016118 | 0.587411 |
| (1,4,1024,1024,64) | 0.146144 | 0.068403 | 0.468053 |
| (1,4,1024,1024,128) | 0.703263 | 0.150958 | 0.214654 |
| (16,12,1024,1024,64) | 5.117440 | 2.225830 | 0.434950 |

四个shape分别满足 `<0.95`。这里沿用共享 harness 的 CUDA-event launch序列计时，每次调用含host侧初始化，GPU等待dispatch的间隙可能被计入；单kernel duration另看NCU。

中间的C02虽然正确，D64的两个1024用例却比02更慢。NCU给出的主要线索是V转置写入shared时约32-way bank conflict，以及普通B iterator读取shared时的冲突。C03只把V改为 `DefaultMmaCore` 的swizzled layout、ThreadMap、shared iterator和128-bit `cp_async`，QK与softmax算术保持不变。

下面三列来自同一 `(1,4,1024,1024,64)`。02和C02取自前一轮已验证采集，C03绑定当时的最终二进制；并非把C02数据当成C03数据。

| NCU metric | 02 | C02 | 最终03 |
|---|---:|---:|---:|
| kernel duration / us | 180.86 | 219.968 | 86.880 |
| L1/TEX throughput / % | 78.35 | 90.62 | 69.56 |
| Tensor active cycles / elapsed cycles / % | 10.4579 | 8.2481 | 21.3110 |
| MIO throttle / cycles per issued instruction | 2.8450 | 5.8238 | 1.1798 |
| registers / thread | 255 | 168 | 168 |

C02→最终03的shared store bank conflicts从4,073,809降到0；shared load conflicts从4,128,768降到2,293,760。V访问的改动得到实测支持，但Q/K仍用普通canonical布局，shared load冲突没有全部消失。这里的Tensor指标是elapsed周期中的活跃比例，不是把FP32操作数换算成Tensor FLOPs的比例。

NCU选择 grid=(16,4,1)、block=(128,1,1) 的真实 `cutlass::Kernel<FlashAttnKernelSplitQ<...>>`；最终03的dynamic shared memory为24 KiB。使用 kernel replay、cache-control=all、clock-control=base，过滤kernel后skip=2、count=1；另外采集SourceCounters与shared memory表。控制时钟下的NCU duration不能替代上面的未profile计时，也不能把单个D64采集推广到所有shape或SM80。

证据和复现参数：

- [候选父子关系与失败记录](../../docs/03-split-q-results.md)。
- [最终验证](../../build/split-q-study/validation/verification.json)、[原始计时](../../build/split-q-study/validation/benchmark.csv)、[完整脚本日志](../../build/split-q-study/validation/workflow.log)。
- [最终NCU采集脚本](../../build/split-q-study/validation/profile/collect.ps1)、[精确命令参数](../../build/split-q-study/validation/profile/03-split-q-command.json)、[NCU报告](../../build/split-q-study/validation/profile/reports/03-split-q.ncu-rep)、[raw CSV](../../build/split-q-study/validation/profile/analysis/03-split-q-raw.csv)。
- [最终source报告](../../build/split-q-study/validation/profile/reports/03-split-q-source.ncu-rep)、[C02诊断报告](../../build/split-q-study/c02/profile/REPORT.md)。

最终binary SHA256：`4a4d82e8e1730a1930d4113def814e8a781bc529f1676d9959958506e9f79480`。完整source hashes见[manifest](../../build/split-q-study/validation/source-manifest.json)。源码支持SM80/SM89 policy，本轮硬件验证只有SM89。

## 后记

<!-- HUMAN: profiling 后补充自己的结论、踩坑和下一步。 -->
