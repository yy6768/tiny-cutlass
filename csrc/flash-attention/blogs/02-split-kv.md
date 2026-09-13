# CUTLASS学习记（八）：Split KV —— 用CUTLASS 重写 FlashAttention

## 前言

- 本篇一定程度上经过重置。（之前的路线是沿着cutlass 41 example 一次到底， 但是后来发现从online softmax 到 xformer 风格的fused kernel（带上flash attention2的版本）实在是太快）

- 所以这篇开始重置：换成 [LeetCUDA 的 flash-attn 路线](https://github.com/xlite-dev/LeetCUDA/tree/main/kernels/flash-attn)，借用cutlass对 MMA PTX（m16n8k16）的包装把 FlashAttention 重新写一遍。

- 这条线的第一个 kernel：Split KV， 对应

  ```
  // Split QKV across MMA(Warps) using naive matmul MMA&Warp tiling policy.
  // case: The layout of 8 MMA(2x4)  [after] kWarpTileSeqLenQxkWarpTileSeqLenK(2x2) -> 32x2,32x2=64x64:
  // |  [64,64]  |    warp_KV 0    |    warp_KV 1    |    warp_KV 2    |    warp_KV 3    |
  // | warp_QP 0 |-- MMA 0,MMA 0 --|-- MMA 2,MMA 2 --|-- MMA 4,MMA 4 --|-- MMA 6,MMA 6 --|
  // | warp_QP 0 |-- MMA 0,MMA 0 --|-- MMA 2,MMA 2 --|-- MMA 4,MMA 4 --|-- MMA 6,MMA 6 --|
  // | warp_QP 1 |-- MMA 1,MMA 1 --|-- MMA 3,MMA 2 --|-- MMA 5,MMA 5 --|-- MMA 7,MMA 7 --|
  // | warp_QP 1 |-- MMA 1,MMA 1 --|-- MMA 3,MMA 2 --|-- MMA 5,MMA 5 --|-- MMA 7,MMA 7 --|
  __global__ void // Q, K, V, O -> [B, H, N, D]
  flash_attn_mma_stages_split_kv_kernel(half* Q, half* K, half* V, half* O, ...);
  ```

- 参考资料：
  - [LeetCUDA: kernels/flash-attn](https://github.com/xlite-dev/LeetCUDA/tree/main/kernels/flash-attn) `mma/basic/flash_attn_mma_split_kv.cu`
  - [[2205.14135\] FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) 原论文第三节
  - CUDA PTX ISA 文档 `mma.sync` / `ldmatrix` 章节

## 从Online Softmax开始说起

- 上一篇说到Online Softmax 把3-pass 降低成2-pass
- 针对Attention 我们不能对各块独立做softmax 然后相加， 借助online softmax 的思想。 我们维护每行的最大值和指数和， 通过缩放系数（也就是最大值的差值缩放）来合并各块。
- 首先针对QKV 的访存进行Tiling。具体在kernel层进行介绍；
- 原文针对score矩阵的每一行维护了 最大值 $\widetilde {m}_{ij}=\operatorname{rowmax}(S*{ij})$ 还有行的所有和$\widetilde\ell_{ij}=\operatorname{rowsum}(\widetilde P_{ij}).$
- 然后还是一样的，根据指数项最大值是否更新计算新的缩放系数，然后更新临时项$\ell_i^{\mathrm{new}}
   =e^{m_i-m_i^{\mathrm{new}}}\ell_i
   +e^{\widetilde m_{ij}-m_i^{\mathrm{new}}}\widetilde\ell_{ij}.$
-  用新的全局基准 $m_i^{\mathrm{new}}$ 重标定旧输出与当前 block 的贡献，并写回 HBM：

![image-20260912230429331](https://metahome-1310941840.cos.ap-guangzhou.myqcloud.com/image-20260912230429331.png)



## Recap: 一个CUTLASS 风格 的 Kernel

- 在先前我们介绍过一个官方的“cutlass 风格”的kernel是怎么实现的
- 现在我们已经有了一个 flash attention的 algorithm 1， 如何包装这个算法到cutlass 架构里呢？
- 下图为GPT 5.6 sol 所绘制， 针对split kv的attention

![device kernel threadblock](https://metahome-1310941840.cos.ap-guangzhou.myqcloud.com/02-device-kernel-threadblock.svg)



## Device：准备参数/入口

当前 `device/flash_attn.cu` 根据架构和 dtype 选择已实例化的配置，由 `device/flash_attn.h` 中`FlashAttn<Kernel>` 完成初始化和启动。

1. 我是SM89架构（虽然其实除了多了FP8运算和SM80没有什么区别……）
2. dtype 是fp16 （从最常见的开始）

关于启动参数的推导， 我运行的机器是4070 Laptop， 在不启动动态shared memory的情况下，这里SRAM的缓存是48KB：
$$
M_{\mathrm{bytes}}=48\times1024=49152,\qquad
M=\frac{M_{\mathrm{bytes}}}{2}=24576.
$$

于是块大小由 $M$ 和 head dimension $d$ 推出：

$$
B_c=\left\lceil\frac{M}{4d}\right\rceil
   =\left\lceil\frac{6144}{d}\right\rceil,\qquad
B_r=\min(B_c,d).
$$

|  $d$ | $B_c$：每块 K/V 的行数 | $B_r$：每块 Q/O 的行数 |
| ---: | ---------------------: | ---------------------: |
|   32 |                    192 |                     32 |
|   64 |                     96 |                     64 |
|   96 |                     64 |                     64 |
|  128 |                     48 |                     48 |

本次案例以 $d=64$ 为例，得到 $B_r=64、B_c=96$。QK 计算的是 $[64,64]\times[64,96]\rightarrow[64,96]$，PV 计算的是 $[64,96]\times[96,64]\rightarrow[64,64]$。

## Kernel：Tiling / Organization

类似标准的[GEMM kernel](https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/include/cutlass/gemm/kernel/gemm.h#L203)， Kernel层入口主要是做CTA 的问题分解并组织， 这个过程主要可以概括成以下的6步。 我们的实现参考了LeetCUDA的[朴素实现](https://github.com/xlite-dev/LeetCUDA/blob/main/kernels/flash-attn/mma/basic/flash_attn_mma_split_kv.cu)

### **Grid / 早退逻辑**

- `blockIdx.x/y/z` 对应 `(Tr, head, batch)`。
  - $T_r$ = 
- 按照problem.size划分后如果超过最大的seq_length或者head_num 就越界了

![Grid 划分](https://metahome-1310941840.cos.ap-guangzhou.myqcloud.com/image-20260913095808652.png)

### Tiling

![image-20260913084939478](https://metahome-1310941840.cos.ap-guangzhou.myqcloud.com/image-20260913084939478.png)

- 按照FA原文计算 $T_r,  B_r, T_c$  的数个小块。

- 对于特定的threadblockIdx:
  - Q 从 `(q_tile × Br, 0)` 开始
  - K/V 从 `(0, 0)` 开始扫描。先偏移到当前 batch/head，再用二维坐标定位 tile；
  - 如何

- 计算Mainloop的循环次数。
  -  这里外层循环是 `ceil(Sk/Bc)` 次 KV tile 扫描。
  - **QK 内部沿 D 的 MMA 循环仍由下层负责**(split-kv)



### 构造组件 Iterator / Mainloop / Epilogue 



1. **构造输入 iterator。** Kernel 用 `Params` 中预计算的 stride、输入指针、extent 和 tile 起点构造 Q/K/V 的 CUTLASS global access iterator。`canonical_warp_idx_sync()` 给出 warp-uniform 的 warp id，用于随后构造 Mma。
2. **进入 Mainloop。** 清零输出累加器和 denominator，把 iterator 与循环次数交给 Mma。shared/warp iterator 依赖 buffer、stage 和 warp 分工，留在 Mainloop 内部构造。
3. **组装 Epilogue。** 定位输出 tile，交给 Epilogue 归一化并写回。当前输出接口仍是 `TensorRef`，还没有接入 stock output tile iterator。

这里有一个影响正确性的细节：stock `PredicatedTileAccessIterator` 先处理 residue，再处理完整 tile。例如 `Sk=65、Bc=64` 时，先读第 0 行，再读第 1–64 行。因此 Mainloop 的 score mask 也必须先按 1 行有效 KV、再按 64 行处理。分块方式改变后仍然覆盖相同的 K/V，但浮点结果需要重新做 reference parity。

## DefaultFlashAttn：组装组件，不重复限制运行时支持范围

`kernel/default_flash_attn.h` 接收 `ArchTag`、元素类型、CTA shape、两个 warp shape、`Stages` 和外部 layout，组装：

```text
LayoutQ/K/P/V
    → LoaderQ/K/V
    → WarpQK / WarpPV
    → FlashAttnMma + FlashAttnEpilogue
    → FlashAttnKernel
```

它是类似 CUTLASS `DefaultXxx` 的 policy factory。SM80/SM89、FP16、BSHD、四个 D 和具体 stage 数都由 device 层显式选择，不再由工厂中的固定形状断言表达。当前 02-split-kv 自己的源码中已经没有 `static_assert`；不支持的运行时配置从 device API 明确返回失败。

## Threadblock：一个 CTA 怎样完成 FlashAttention

### CTA：固定 Q，流式扫描 K/V

`blockIdx.x/y/z` 对应 `(q_tile, head, batch)`。同一个 head 的行 stride 是 `H×D`，所以即使 Q/K/V 是 BSHD，单 head 的二维视图也不是紧密排列的矩阵。Kernel 层用这个 stride 构造 global iterator，Mainloop 接收并推进 iterator，loader 负责把当前 tile 异步搬到 shared memory。

Q tile 只加载一次。每轮读取 K/V tile，得到：

$$
S_j=Q_iK_j^T/\sqrt D,\qquad
m_j=\max(m_{j-1},\operatorname{rowmax}S_j),\qquad
\alpha_j=e^{m_{j-1}-m_j}
$$

$$
P_j=e^{S_j-m_j},\qquad
l_j=\alpha_jl_{j-1}+\operatorname{rowsum}P_j,\qquad
O_j=\alpha_jO_{j-1}+P_jV_j.
$$

所有 KV tile 结束后才计算 `O/l`。初始化是 `m=-∞, l=0, O=0`，因此第一轮的 α 为 0。当前 iterator 先处理 residue tile，它的无效 score 列在 softmax 前置为 `-∞`，对应 P 为 0；无效 Q 行参与 CTA 的同步，但不写回输出。

![online rescale](assets/02-recomputation.svg)

图中对旧输出做的是 **rescale**：把已有的 O/l 换到新最大值对应的指数基准上，并不重算以前的 QK。我们保留的 O 还是未归一化分子。

### Warp 协作：QK 切成 2×4，PV 重新解释列方向

![warp split](assets/02-warp-split-kv.svg)

QK 的每个 warp 算 `[32,16]`，由 2×2 个 `m16n8k16` 输出块组成。`warp_m=warp_id%2`，`warp_n=warp_id/2`。四个 `warp_n` 各持有一行的 16 个 score，没有一个 warp 能独自完成整行 softmax。

到了 PV，同一 warp 的行仍然对应原来的 32 行 Q，但列改为输出通道 Dv。不能把 QK 的 `warp_n` 继续当成 KV 分片；P 必须先写到 shared memory，让每个 PV warp 读到完整的 64 列 P。

这也解释了 split-KV 的通信代价：每轮需要交换四份 row max、四份 row sum，还需要把 P 从 QK 的寄存器分布重排为 PV 的输入布局。这里没有专门的 reducer warp：相同 `warp_m` 的四个 warp 在 barrier 后各自读取四份统计量，所以 running max 和 denominator 在四个列 warp 中重复保存。

### CTA 内同步：2-stage 时序与 barrier

[FlashAttnMma](../02-split-kv/threadblock/flash_attn_mma.h) 持有整条 attention 主循环。device 当前选择 `Stages=2`：prologue 异步加载 Q 和 K0，`wait<0>` 后用一次 CTA barrier 让它们对所有 warp 可见。每个 KV tile 按下面的次序运行：

1. **issue Vj 和 K(j+1)**：V 写入唯一的 V buffer；若还有下一块 K，则写入另一个 K stage。这里只 commit group，不立即等待。
2. **QK → local max → barrier**：当前 Q/K 已在 prologue 或上一轮末尾准备好。每个 quad 的 lane 0 写 `row_max[warp_n][row]`，四个列 warp 交换最大值。
3. **更新 m/α → exp → P store 与 local sum → barrier**：P 以 FP16 写入 shared，row sum 用 FP32 scratch 交换。
4. **等待 V → barrier → PV**：`wait<1>` 留下最新的 K prefetch，只等待更早的 V group；最后一个 tile 使用 `wait<0>`。CTA barrier 把每个线程的完成状态变成 collective shared 可见性。
5. **PV 后 barrier**：所有 warp 离开 P/V，下一轮才可以覆盖单 buffer 的 P/V 和 row statistics。
6. **等待下一 stage K → barrier**：`wait<0>` 后翻转 `read_stage`，下一轮才读取新 K。

当前代码是 prologue 一次，再加每个 KV tile 五次 CTA barrier。row statistics 在不同列 warp 间有冗余，但每份 O 都需要相同的 m/l 才能独立完成归一化。上游因为采用单独的 CTA-wide second reduction，源码字面上的 barrier 更多；这里保留等价的数据依赖，但用 CUTLASS fragment 映射让消费 warp 自己合并四项。

![two-stage async mainloop and barriers](assets/02-mainloop-barriers.svg)

这些 barrier 保护的是 shared-memory 的可见性和复用，而不是跨 CTA 协作。`q` 在 prologue 后一直存活；`k[0]/k[1]` 交替读写；`v/p/row_max/row_sum` 每轮覆盖；O、running max 和 denominator 留在每个 warp 的寄存器 fragment 中。当前 `SharedStorage` 也不是 union：Q、两个 K stage、V、P 和两组统计量同时占空间，epilogue 没有 shared scratch，所以没有必要伪造 mainloop/epilogue 的 union 复用。

P 转为 FP16 后才进入 Tensor Core，而 l 是未舍入的 FP32 P 的和。这会引入小的数值误差；本版依靠实测 tolerance 验证，并没有宣称与 FP32 softmax 的数学表达逐 bit 相等。上游 basic 源码的部分 QK 路径使用 half accumulator，本版 QK/PV 都使用 FP32 accumulator，不能直接把两个版本的精度和速度视为同一个实现。

### 这是什么 pipeline

当前实现是 **K 方向的 2-stage async pipeline**，但不要把它理解成 CUTLASS GEMM 在 head-dim K-loop 内部的 `MmaMultistage`：

```text
Q: [single persistent buffer]
K: [stage 0] [stage 1]   <- next KV tile prefetch
V: [single buffer]       <- copy overlaps current QK/softmax
P: [single buffer]
```

`FlashAttnTileLoader::copy()` 用 `PredicatedTileAccessIterator` 计算带边界保护的 global 地址，用 `RegularTileAccessIterator` 计算 swizzled shared 地址，再发出 16-byte `cp_async_zfill`。无效的 sequence row 或 padding channel 由 zfill 置零。`cp_async_fence()` 划分 group，`cp_async_wait<N>()` 控制最多留下多少个未完成 group。

stage 数只描述 K buffer。V 没有双缓冲，只是它在 QK/softmax 期间异步飞行；所有 8 个 warp 都既发 copy 又做计算，没有 producer/consumer warp specialization。warp MMA 内部的 head-dim 循环仍按 K=16 顺序执行 `load → transform → mma.sync`，没有 fragment ping-pong。因此这是一条 KV-tile pipeline，而不是直接复用标准 GEMM 的 K-loop pipeline。

### Tile loader：layout 与 warp iterator 是一组契约

[FlashAttnTileLoader](../02-split-kv/threadblock/tile_loader.h) 用 `PitchLinearWarpRakedThreadMap` 分配每个线程的 128-bit 访问。它不再经过 register fragment：global access iterator 与 shared access iterator 直接给出源/目标地址，`cp_async_zfill<16>` 完成带 predicate 的 global→shared copy。

| 数据 | Shared layout | 逻辑视图 |
|---|---|---|
| Q | RowMajor TensorOp Crosswise `<16,64>` | `[64,pitchQ]` |
| K | ColumnMajor TensorOp Crosswise `<16,64>` | `[pitchQ,64]`，作为 QK 的 B |
| P | RowMajor TensorOp Crosswise `<16,64>` | `[64,64]`，作为 PV 的 A |
| V | RowMajor TensorOp Congruous `<16,64>` | `[64,pitchV]`，作为 PV 的 B |

K 在 global memory 中仍是 sequence 行、D 列。loader 沿 D 连续读取，存入的 shared memory 从 column-major 视图看就是 Kᵀ，不需要单独运行 transpose kernel。

这里踩到的坑是：**能实例化 CUTLASS 类型，不代表任意 warp tile 都满足它内部的地址变换。** 当前版本的 congruous B 迭代器在 N=32 时会置换内部指针，但 N=16 的 `add_tile_offset()` 没有对应处理；N=8 还会使完整 ldmatrix 的迭代数变成 0。直接照搬上游 `Dv/4` 的 warp 宽度会得到错误结果。

本版保留四个 PV 列 warp，统一令每 warp N=32，将 V/PV 的内部通道宽度补齐到 128，epilogue 只写有效 D 列。Q/K 的 shared pitch 则向上补齐到 64 的倍数，QK 只循环真实的 D/16 次。于是：

| D | pitchQ | pitchV | PV 每 warp tile | 每 CTA dynamic shared memory |
|---:|---:|---:|---|---:|
| 32 / 64 | 64 | 128 | 32×32 | 51,200 B |
| 96 / 128 | 128 | 128 | 32×32 | 75,776 B |

这是一项明确的首版代价：D=32/64/96 都存在无用 PV 通道计算。后续可以调整 PV warp 分工，或实现经过独立验证的窄 N iterator；不能只把模板参数改小就认为优化成立。

## Warp MMA：fragment 不是线性的二维数组

[FlashAttnWarpMma](../02-split-kv/warp/flash_attn_mma.h) 每次加载 A/B fragment，调用 `mma.transform()` 后执行 MMA，再推进到下一个 K=16 的 instruction group。`transform()` 属于 CUTLASS 的 operand 契约，不能因为元素类型没有变化就省略。

softmax 要知道每个 accumulator 对应哪一行。这里使用的 CUTLASS `MmaTensorOp` 默认 `AccumulatorsInRowMajor=false`：逻辑输出虽是 row-major，instruction tile 在 fragment 中按 `m+n*MmaIterations::kRow` 排列。单个 m16n8 的四个 float 中，前两个对应一行的相邻两列，后两个对应行号加 8。

因此 `row_slot(i)`、`row(slot,lane)` 和 `column(i,lane)` 专门解释这个 fragment。4 个相邻 lane 组成一个 quad，覆盖同一行的 8 列，用两次 XOR shuffle（1、2）归约。这个映射同时用于 score 归约、P shared store、O rescale 和最终输出；换 MMA 类型或 accumulator 排列时必须一同检查。

## Epilogue：归一化并写回真实 D 列

`FlashAttnEpilogue` 不申请 shared scratch。每个 PV warp 已经拥有自己负责的输出通道 accumulator，因此它只读取对应行的 denominator，计算 `accum/l`，转换成 FP16，并对 sequence tail 与补齐到 128 的无效输出列做 predicate store。

这与标准 CUTLASS GEMM epilogue 还有距离：当前没有 output tile iterator，也没有向量化 collective store，而是逐 fragment element 写回。它是清晰的首版职责边界，不应被描述成已经复用了完整 CUTLASS epilogue machinery。

## 验证

Kernel 入口改为传入 global iterator 后，在 RTX 4070 Laptop GPU（SM89）重新构建了 `flash_attention_test` 和 `split_kv_attention`，并运行 cuDNN SDPA parity：18 组 02-split-kv case 覆盖 D=32/64/96/128、Sq/Sk tail、`Sq≠Sk` 与不同输入 scale；另有 4 组 `--kernel=all` 回归和 2 组非法 D/Dv 拒绝，全部通过。D=96、Sq=65、Sk=257 的多 tile/tail case 还通过 compute-sanitizer memcheck、racecheck 和 synccheck，均无错误。当前验证记录和 sanitizer 日志在 `build/split-kv-study/kernel-entry-refactor/`。

## 源码对照

- 算法、warp 切分与 stage 时序：[LeetCUDA split-KV，固定 revision](https://github.com/xlite-dev/LeetCUDA/blob/0983c6554c39a18c1ae53739030090d0db127338/kernels/flash-attn/mma/basic/flash_attn_mma_split_kv.cu)。本版不复制其 PTX 宏或 Torch ownership，而是用 CUTLASS 组件表达相同的数据依赖。
- 组装方式：[CUTLASS example 13](../../../3rdparty/cutlass/examples/13_two_tensor_op_fusion/kernel/default_b2b_gemm.h)。
- Warp MMA 与 fragment 排列：[mma_tensor_op.h](../../../3rdparty/cutlass/include/cutlass/gemm/warp/mma_tensor_op.h)。
- Ldmatrix iterator 与 tile offset：[mma_tensor_op_tile_iterator.h](../../../3rdparty/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator.h)。
- Async global iterator：[predicated_tile_access_iterator.h](../../../3rdparty/cutlass/include/cutlass/transform/threadblock/predicated_tile_access_iterator.h)。
- Shared access iterator：[regular_tile_access_iterator_tensor_op.h](../../../3rdparty/cutlass/include/cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op.h)。
- `cp.async` 包装：[memory_sm80.h](../../../3rdparty/cutlass/include/cutlass/arch/memory_sm80.h)。
- 本地 CUTLASS 4.5.2，revision `1732ed7da3b81d9f28b0130370ba70755a7e6dda`。

## Profile

这一轮改变了 global iterator 的构造与推进方式，旧 NCU 数据不再代表当前二进制。通过验证后已用 `bench.py` 重新计时，记录在 `build/split-kv-study/kernel-entry-refactor/benchmark.csv`；当前尚未重新采集 NCU，不在这里推断性能变化的原因。

## 后记

<!-- HUMAN: profiling 后补充自己的结论、踩坑和下一步。 -->
