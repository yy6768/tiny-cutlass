# CUTLASS example 41 (fused MHA) 在 SM89(Ada) 上的组件调用栈与寄存器/共享内存布局迁移

> 分析对象：`3rdparty/cutlass/examples/41_fused_multi_head_attention/kernel_forward.h`
> 目标配置：`scalar_t = cutlass::half_t`，`ArchTag` 实例化到 `cutlass::arch::Sm80`
> （见下文「为什么 SM89 用的是 Sm80 分支」），`kQueriesPerBlock = kKeysPerBlock = 64`，
> 单 warp 负责一个 32x32 累加器 tile（`WarpShape = 32x32`, `InstructionShape = 16x8x8`）。

配套可视化图放在 `blogs/assets/03-mha-*.excalidraw`（用 Excalidraw 打开，或用
`blogs/assets/gen_mha_diagrams.py` 重新生成/编辑）。

---

## 0. 为什么 SM89(Ada) 走的是 `arch::Sm80` 这条路径

`kernel_forward.h` 本身不做 arch dispatch，dispatch 发生在调用它的 host 代码里
（xformers 的 `gemm_kernel_utils.h::DISPATCH_ARCHTAG`）：

```cpp
#define DISPATCH_ARCHTAG(CC, func)                     \
  {                                                     \
    if (CC >= 80) {                                     \
      using ArchTag = cutlass::arch::Sm80;               \
      func();                                            \
    } else if (CC >= 75) { ... }                          \
  }
```

SM89 的 compute capability 是 89，满足 `CC >= 80`，于是 `ArchTag = cutlass::arch::Sm80`。
example 41 里**没有**为 SM89/SM90 单独特化过 `AttentionKernel`，所以在 Ada 上实例化出来的
`MM0::Mma` / `MM1::Mma` 和 A100（Sm80）完全同构：同样的 `MmaMultistage`、同样的
`m16n8k8`/`m16n8k16` tensor-op 指令、同样的 `cp.async` 流水线深度选择规则
（`kIsHalf ? 4 : DefaultConfig::kStages`）。SM89 相对 SM80 新增的 fp8 tensor core
（`mma_sm89.h` 里的 `OpMultiplyAdd` fp8 特化）在这份 example 里完全没有被触达 —— 这也是
`tiny-cutlass` 当前 SM89 FP8 迁移工作和这份 example 的关系：example 41 是 SM80 fp16 融合
attention 的参考实现，SM89 fp8 需要另外的 `DefaultGemmType` 特化和新的 `B2bGemm`/
`WarpIteratorFromSmem` 数据类型分支，不能指望改一下 `ArchTag` 就自动获得 fp8 加速。

---

## 1. 顶层结构与调用栈总览

```
attention_kernel_batched<AK>(Params)                       [kernel_forward.h:1312]  __global__
  └─ AK::Params::advance_to_batch()                          [kernel_forward.h:233]   host-precomputed strides 的运行期分支
  └─ AK::attention_kernel(Params&)                           [kernel_forward.h:652]   本文主线
       ├─ MM0::Mma  mma(...)                                 见 §2
       │    └─ mma(gemm_k_iterations, accum, iterA, iterB, accum)   —— Q @ K^T
       ├─ [可选] bias 加载 + accum += bias                     TileSmemLoader (transform/tile_smem_loader.h)
       ├─ [可选] causal mask                                   AccumLambdaIterator::iterateRows
       ├─ iterative_softmax<IteratorC>(...)                   [kernel_forward.h:1153]  见 §3
       ├─ MM0::B2bGemm::accumToSmem(...)                       [mma_from_smem.h:1584]   见 §4
       ├─ [可选] dropout（curand，按行连续消费随机数）
       ├─ MM1::Mma mma_pv(...)                                 见 §5（含 §4.1 prologue）
       │    └─ mma_pv(gemm_k_iterations, accum_o, iterV, accum_o)   —— P @ V
       └─ Epilogue（EpiloguePipelined + MemoryEfficientAttentionNormalize）  见 §6
```

`AttentionKernel` 是一个模板类，`MM0`/`MM1` 是它内部的两个「装配」结构体（不是运行时对象，
是编译期用来选类型的 namespace-like struct）。理解这份代码的关键在于：**MM0 和 MM1 各自
通过 `DefaultMma` / `FindDefaultMma` / `DefaultMmaFromSharedMemory` 这套工厂模板，组装出
一整条 Iterator → SmemIterator → WarpIterator → Operator(MMA) → Epilogue 的流水线**，
这条流水线在两次矩阵乘法之间通过共享内存（而不是 global memory）传递中间结果 `P = softmax(QK^T)`。

---

## 2. MM0：`Q @ K^T`，模板装配点

文件：`kernel_forward.h:387-468`（`struct MM0`）

| 别名 | 装配自 | 来源文件 |
|---|---|---|
| `GemmType` | `gemm_kernel_utils::DefaultGemmType<Sm80, half>` | `gemm_kernel_utils.h:177` — 选出 `OpClassTensorOp` + `InstructionShape=16x8x8` + `OpMultiplyAdd`，`ThreadK=WarpK=32` |
| `DefaultMma` | `FindDefaultMma<half,RowMajor,...,half,ColumnMajor,...>::DefaultMma` | `gemm/find_default_mma.h:89` — 对 Sm80+fp16 走通用 `cutlass::gemm::threadblock::DefaultMma`（不是 fp32 FastF32 的特殊分支，那条分支只对 `kAlignmentA>1` 且 arch==Sm80 且 ElementA 走 fp32 才触发） |
| `MmaCore` | `DefaultMma::MmaCore` | `cutlass/gemm/threadblock/default_mma_core_sm80.h`（3rdparty，未改写） |
| `IteratorA`/`IteratorB` | `DefaultMma::IteratorA/IteratorB` | 全局内存 → 共享内存的 predicated tile iterator，来自 3rdparty `cutlass/transform/threadblock/predicated_tile_iterator.h` |
| `DefaultThreadblockMma` | `DefaultMma::ThreadblockMma` | `cutlass::gemm::threadblock::MmaMultistage`（fp16 时 `kStages=4`，见 `ArchTag::kMinComputeCapability>=80 && kIsHalf ? 4 : DefaultConfig::kStages`） |
| `Mma` | `kSingleValueIteration ? MakeCustomMma<...>::Mma : DefaultThreadblockMma` | `gemm/find_default_mma.h` 的 `MakeCustomMma` 特化 —— 当 `kMaxK <= kKeysPerBlock`（即 head_dim 能在一个 key-block 内完整算完）时，包一层自定义 mainloop 以省掉尾部 residual 分支 |
| `AccumLambdaIterator` | `DefaultMmaAccumLambdaIterator<Mma::Operator::IteratorC, float, 32>::Iterator` | `gemm/mma_accum_lambda_iterator.h:361` — Sm80 tensor-op 累加器的「逐元素回调」迭代器，本文 §3 大量依赖它 |
| `BiasLoader` | `TileSmemLoader<half, MatrixShape<64,64>, ...>` | `transform/tile_smem_loader.h` — attn_bias 从 global 搬到 shared 的 helper |
| `B2bGemm` | `cutlass::gemm::threadblock::B2bGemm<Mma::Operator::IteratorC, Mma::Operator, half, WarpShape, ThreadblockShape>` | `gemm/mma_from_smem.h:1469` — **本文 §4 的主角**，MM0 输出寄存器 → smem 的桥梁 |
| `AccumulatorSharedStorage` | `B2bGemm::AccumulatorSharedStorage` | 同上文件 `mma_from_smem.h:74-120` — smem 里存放 `Si=P` 的 buffer 类型 |

`Mma` 实际类型链：`DefaultGemmType`（选指令形状/算子）→ `FindDefaultMma`（选 Sm80 通用
`DefaultMma` 工厂）→ `cutlass::gemm::threadblock::DefaultMma`（3rdparty 标准工厂，内部再调
`DefaultMmaCore`）→ 最终得到 `MmaMultistage<ThreadblockShape<64,64,32>, ..., 4 stages>`。

---

## 3. Softmax：寄存器内部化归约 + 跨 warp 共享内存归约

函数：`iterative_softmax<WarpIteratorC>(...)`，`kernel_forward.h:1153-1298`。

核心依赖 `AccumLambdaIteratorSm80`（`gemm/mma_accum_lambda_iterator.h:47-117`），它把
「TensorCore 累加器 fragment 的第 `idx` 个寄存器元素对应矩阵里的 `(accum_m, accum_n)`」
这件事封装成一个 `iterateRows(lane_offset, beginRow, op, endRow)` 回调接口，屏蔽了不同架构
（Sm70/Sm80/Simt）累加器布局的差异。

三段归约，全部发生在**同一批寄存器**上，靠 `__syncthreads()` 和 `atomicMaxFloat` /
`__shfl_xor_sync` 衔接：

1. **行最大值 `mi[row]`**：每行的 4 个 lane（同 `quad`，见图2）各自算出自己持有元素里的
   max，再用 `atomicMaxFloat(&mi[accum_m], max)` 写共享内存 `ScalingCoefs::mi`
   （`kernel_forward.h:554-561` 定义在 `SharedStorage` 里）。用 atomic 而不是先做 warp 内
   规约是因为**同一行跨了 4 个 lane 但这 4 个 lane 不在同一个 `mma_n` tile 里**（不同 lane
   持有不同列），4 lane 各自还要遍历多个 `mma_n`，所以作者选择「每个 (lane, mma_n) 组合都
   直接 atomicMax 进共享内存」而不是先做同 quad 内的 shuffle 规约。
2. **`m_prime`/`out_rescale` 更新**：`thread_id() < kQueriesPerBlock` 的线程各自负责一行，
   比较新旧最大值决定 rescale 因子，处理 `kSupportsBias` 场景下全 `-inf` 行的 NaN 保护
   （`restore_mi_to_minus_inf`）。
3. **行求和 `s_prime[row]`**：先对每个 `(row, mma_n)` 内部求和（寄存器内简单累加），再用
   `LambdaIterator::reduceSameRow` 做 `__shfl_xor_sync(1)` 再 `__shfl_xor_sync(2)` 的蝴蝶规约
   （同 quad 4 个 lane 两两配对再交叉），`lane_in_quad==0` 的线程把结果写进
   `addition_storage[accum_m + kQueriesPerBlock*tile_offset.column()]`
   （`ScalingCoefs::addition_storage`），最后 `thread_id()<kQueriesPerBlock` 的线程再循环
   `MmaCore::WarpCount::kN` 次把同一行、不同 warp-tile-column 的部分和累加进 `s_prime`。

可视化：`blogs/assets/03-mha-softmax-reduction.excalidraw`。

---

## 4. MM0 Epilogue：`B2bGemm::accumToSmem`（寄存器 → 共享内存，丢失 lane 身份）

文件：`gemm/mma_from_smem.h:1469-1634`（Sm80 tensor-op 特化）。

调用点：`kernel_forward.h:908-910`

```cpp
MM0::B2bGemm::accumToSmem(
    shared_storage.after_mm0.si, accum, my_lane_id, output_tile_coords);
```

内部管线：

```
accum (FragmentC, per-lane 寄存器, float)
  └─ FragmentIteratorTensorOp<WarpShape,InstructionShape,accum_t,FragmentC,RowMajor>
       // 把 "TensorCore 累加器寄存器排布" 重新解释为一系列可顺序读出的小 fragment
  └─ EpilogueSmemAccumulator<SmemIteratorD0, FragmentIteratorAccumulator, ..., OutputOpNoOp>
       // OutputOpNoOp = LinearCombination<half, ..., ScaleType::Nothing>：只做 float->half 转换，不做数值变换
  └─ TileIteratorTensorOp<WarpShape,InstructionShape,half,RowMajor>  (= SmemIteratorD0)
       // 以 "ldmatrix 友好" 的方式把数据写成 **行主序** 共享内存布局
  └─ AccumulatorSharedStorage.accum (即 shared_storage.after_mm0.si)
```

关键点（也是图3要传达的核心）：**这一步之后，数据的寻址方式从「按 lane_id + 寄存器下标」
变成「按 (row, col) 行主序共享内存地址」，per-lane 身份完全消失**。这块共享内存是
`SharedStorageAfterMM0` 里的一个 union 成员（`kernel_forward.h:566-568`），和 `BiasLoader`
的 smem 复用同一块地址（因为 bias 只在写 `Si` 之前用一次）。

`AccumulatorSharedStorage`（`mma_from_smem.h:79-120`）就是一个 `AlignedBuffer<half, ...>`
包一个 `packed RowMajor layout`，本质上和普通 GEMM 的 smem staging buffer没有区别 —— 这也是
为什么它能直接被 MM1 当作「从 smem 读 operand A」的输入。

---

## 5. MM1：`P @ V`，装配点 + smem-resident operand A

文件：`kernel_forward.h:470-547`（`struct MM1`）。

| 别名 | 装配自 | 来源文件 |
|---|---|---|
| `DefaultGemm` | `cutlass::gemm::kernel::DefaultGemm<half,RowMajor,...>` | 3rdparty `cutlass/gemm/kernel/default_gemm.h` — 这里只是借用它产出标准 `Mma`/`Epilogue` 类型给后续覆盖用，本身不被直接实例化执行 |
| `WarpIteratorA` | `DefaultWarpIteratorAFromSharedMemory<WarpShape, InstructionShape, DefaultGemm::Mma::Policy::Operator::IteratorA, Policy>::WarpIterator` | `iterators/default_warp_iterator_from_smem.h:58-75` — Ampere fp16 分支选中 `cutlass::gemm::warp::WarpIteratorFromSmem<kA, half, MatrixShape<16,8>>` |
| `DefaultMmaFromSmem` | `DefaultMmaFromSharedMemory<DefaultGemm::Mma, kMaxK, WarpIteratorA, false>` | `gemm/mma_from_smem.h:1270-1465` — 把「标准 MmaMultistage」改造成「operand A 从共享内存读，而不是从 global memory 走 SmemIteratorA」的版本 |
| `Mma` | `DefaultMmaFromSmem::Mma` = `MmaMultistageFromSharedMemory<...>` | `gemm/mma_from_smem.h:696-1267` |
| `IteratorB` | `Mma::IteratorB` | 标准 global→smem iterator，加载 `V` |
| `DefaultEpilogue`/`OutputTileIterator*` | `DefaultGemm::Epilogue` 及其 `PredicatedTileIterator` 特化 | 用于最终写回 global memory |

### 5.1 MM1 prologue：`WarpIteratorFromSmem`（smem → 全新寄存器）

文件：`iterators/warp_iterator_from_smem.h`。这是**图4**的主角，也是最容易被误解的一步：
很多人会以为 MM0 输出寄存器直接「变形」成了 MM1 输入寄存器，但实际上中间经过了一次完整的
**smem 落地 + `ldmatrix` 重新加载**，两次的 per-lane 归属规则完全不同，互相独立：

- MM0 累加器（图1）：`accum_m = mma_m*16 + row*8 + quad`，`accum_n = mma_n*8 +
  lane_in_quad*2 + col`，来自 tensor-core **累加器**寄存器的硬件规范。
- MM1 operand-A（图4右）：`origin = (lane_id % 8, 0)`，再叠加
  `ldsm_vec_num = lane_id >> 3` 驱动的 `(access_m_idx*8 + inst_m_idx*16, inner_idx*32)`
  偏移 —— 这是 `ldmatrix.sync.aligned.x4.m8n8.shared.b16` 指令**本身**的硬件加载规范
  （8 个连续 lane 一组，每组发一条 `ldmatrix.x4`，一次搬 4 个 8x8 sub-tile）。

两者唯一的联系是：它们描述的是**同一块共享内存 `Si` 里的同一份数值**，布局转换的「翻译」
工作完全由中间的行主序共享内存完成 —— 寄存器→smem 和 smem→寄存器 是两次独立的、各自遵循
自己硬件规则的映射，softmax 归约之所以必须先落地到 smem 再重新加载，正是因为 MM0 的累加器
布局和 MM1 需要的 `ldmatrix` 操作数布局**不兼容**，没有办法用 warp 内 shuffle 直接互转。

`kTranspose` 参数（本例中 `kScaleOperandA=false`，且是否 transpose 由
`DefaultMmaFromSharedMemory` 里的 `WarpIteratorTranspose` 决定，仅在 `MmaMultistage`
分支且用户要求转置 A 时生效；本配置未启用）。

### 5.2 MM1 mainloop 与 prologue 时机

`prologueV`（`kernel_forward.h:732-744`）在 `kPreloadV`（Sm80 + fp16 时为真）时，
**在 MM0 的 mma 完成、`accum` 还在寄存器里的时候**就已经发起了 V 的 `cp.async` 预取
（`kernel_forward.h:802-806`），与 softmax、`accumToSmem`、dropout 并行执行，隐藏 V 的
global memory 延迟。这是 example 41 相对于「先做完 softmax 再天真地发起下一次 GEMM」的
关键性能优化点之一。

`mma_pv(gemm_k_iterations, accum_o, iterator_V, accum_o)` 调用的是
`MmaMultistageFromSharedMemory::operator()`（`mma_from_smem.h:1019-1266`），mainloop
结构与标准 `MmaMultistage` 几乎一致，区别只在 operand A 走 `WarpIteratorA1_`
（从 smem 读，无需 `cp.async`）而不是 `SmemIteratorA`（从 global 搬）。

---

## 6. Epilogue：`accum_o` → 归一化输出

`kernel_forward.h:1093-1149`（`kKeepOutputInRF` 分支，本配置 `kSingleValueIteration=true`
时走这条路，因为 `head_dim_value <= kKeysPerBlock`）：

```
accum_o (FragmentC, per-lane寄存器)
  └─ EpiloguePipelined<..., MemoryEfficientAttentionNormalize, ...>
       // alpha = isLast ? 1/s_prime[row] : 1；beta = alpha * m_prime[row]
       // D = alpha*accum_o + beta*source   (source = 上一轮已写的部分结果, 用于跨 key-tile 累加场景)
  └─ OutputTileIterator → global memory (output_ptr)
```

若 `!kKeepOutputInRF`（多值迭代场景，`head_dim_value > kKeysPerBlock`），则每个 key-tile
迭代末尾都要跑一次这个 epilogue，写入 `output_accum_ptr`（float32 暂存），并且
**下一次 key-tile 的 `iterative_softmax`** 会对已经写回寄存器的 `frag_o` 做
`frag_o[idx] *= out_rescale[accum_m]` 就地 rescale（`kernel_forward.h:1244-1253`），这是
FlashAttention 在线 softmax 数值稳定性的标准做法：每次刷新行最大值后，用比值重新缩放已经
累积的部分输出。

`accum_o` 的累加器布局与图1（MM0 累加器）**完全同构**（同为 Sm80 tensor-op accumulator,
`IteratorC` 同一个模板特化），只是矩阵维度换成了 `kQueriesPerBlock x head_dim_value`。
可视化：`blogs/assets/03-mha-mm1-accum-layout.excalidraw`。

---

## 7. 完整文件清单（按被引用顺序）

| 文件 | 角色 |
|---|---|
| `kernel_forward.h` | 顶层 kernel、`Params`、`MM0`/`MM1` 装配、softmax、主循环控制流 |
| `gemm_kernel_utils.h` | `DefaultGemmType`（指令形状/算子选择）、`DISPATCH_*` 宏、`warp_uniform` |
| `gemm/find_default_mma.h` | `FindDefaultMma` —— MM0 的 Mma 工厂封装（覆盖 kStages） |
| `gemm/mma_accum_lambda_iterator.h` | `AccumLambdaIteratorSm70/Sm80/Simt` —— 累加器 (idx ↔ (row,col)) 映射，softmax 依赖它 |
| `gemm/mma_from_smem.h` | `AccumulatorSharedStorage`、`MmaPipelined/MultistageFromSharedMemory`、`DefaultMmaFromSharedMemory`、`B2bGemm`（本文 §4/§5 核心） |
| `iterators/warp_iterator_from_smem.h` | `WarpIteratorFromSmem` —— ldmatrix 风格 smem→寄存器 A 操作数加载（§5.1） |
| `iterators/default_warp_iterator_from_smem.h` | 按架构/dtype 选择 `WarpIteratorFromSmem` 还是标准 `MmaTensorOpMultiplicandTileAccessIterator` |
| `iterators/transpose_warp_iterator.h` | `TransposeWarpIterator` —— 按需把 A 操作数迭代器换成转置版本 |
| `iterators/make_residual_last.h` | 把 iterator 包装成「最后一个 k-tile 走 residual 边界检查」的版本 |
| `gemm/custom_mma_base.h` / `custom_mma_pipelined.h` / `custom_mma_multistage.h` / `custom_mma.h` | `MakeCustomMma` —— MM0 在 `kSingleValueIteration` 时替换标准 mainloop |
| `transform/tile_smem_loader.h` | `TileSmemLoader` —— attn_bias 的 global→shared 搬运 |
| `epilogue/epilogue_pipelined.h` | `EpiloguePipelined`（支持 row_id 传给 OutputOp 的定制 epilogue） |
| `epilogue/epilogue_rescale_output.h` | `MemoryEfficientAttentionNormalize` —— 最终/中间输出的 rescale 算子 |
| `epilogue/epilogue_thread_apply_logsumexp.h` | backward 用的 LSE 相关 epilogue（forward 主路径未使用，`B2bGemm::accumApplyLSEToSmem` 才用得到） |
| `debug_utils.h` | 调试宏，不影响数据流 |

3rdparty 标准组件（未在此目录改写，但被大量依赖）：
`cutlass/gemm/threadblock/default_mma.h`、`default_mma_core_sm80.h`、`mma_multistage.h`、
`mma_pipelined.h`、`cutlass/gemm/kernel/default_gemm.h`、
`cutlass/epilogue/threadblock/epilogue_smem_accumulator.h`、
`cutlass/gemm/warp/mma_tensor_op_fragment_iterator.h`、
`cutlass/transform/threadblock/predicated_tile_iterator.h`。

---

## 8. 可视化图索引（`blogs/assets/`）

| 文件 | 内容 |
|---|---|
| `03-mha-mm0-accum-layout.excalidraw` | 图1：MM0 累加器寄存器布局（`quad`/`lane_in_quad` 着色） |
| `03-mha-softmax-reduction.excalidraw` | 图2：行最大值 atomicMax 归约 + 行求和蝴蝶 shuffle 归约 → `s_prime` |
| `03-mha-epilogue-to-smem.excalidraw` | 图3：`accumToSmem` —— 寄存器（有 lane 身份）→ 行主序共享内存（无 lane 身份） |
| `03-mha-mm1-prologue-ldmatrix.excalidraw` | 图4：`WarpIteratorFromSmem` —— smem → MM1 操作数 A 的全新寄存器（ldmatrix 硬件分布，与图1不同构） |
| `03-mha-mm1-accum-layout.excalidraw` | 图5：MM1 累加器 `accum_o`（与图1同构，维度换成 query x head_dim_value） |

生成脚本：`blogs/assets/gen_mha_diagrams.py`（Python，用仓库外的
`excalidraw-diagrams` skill 库）；可编辑该脚本重新生成，或直接在
[excalidraw.com](https://excalidraw.com) 打开 `.excalidraw` 文件手改。

### 图1/图5 布局公式（`AccumLambdaIteratorSm80`）

```
quad         = lane_id >> 2         // 0..7
lane_in_quad = lane_id & 3          // 0..3
accum_m = mma_m * InstructionShape.M * OpDelta.row + row * 8 + quad        // row in {0,1}
accum_n = mma_n * InstructionShape.N * OpDelta.col + lane_in_quad*2 + col  // col in {0,1}
```
每个 lane 持有 8 个元素（2 个 row-sub x 4 个 mma_n 位置，图中只画了 mma_n∈{0,1} 的窗口）。
**同一个 quad 的 4 个 lane 持有同一行、不同列** —— 这正是 softmax 行归约（图2）需要跨这 4
个 lane 协作的原因。

### 图4 布局公式（`WarpIteratorFromSmem`，Operand=A, half, InstructionShape 16x8x16）

```
ldsm_vec_num = lane_id >> 3                        // 0..3，四组，每组 8 个 lane
access_m_idx = ldsm_vec_num % kTilesPerInstruction  // kTilesPerInstruction = 2
inst_m_idx   = ldsm_vec_num / (kTilesPerInstruction * kAccessesInner)
origin = (lane_id % 8 + access_m_idx*8 + inst_m_idx*16, inner_idx*32)
```
每组 8 个连续 lane 共享一条 `ldmatrix.sync.aligned.x4` 指令，一次性把 4 个 8x8 sub-tile
（跨 2 个 16-row instruction-tile）从共享内存搬进寄存器 —— 这是硬件指令规定的分布，与图1
的 tensor-core 累加器分布**没有任何直接对应关系**，两者之间必须经过 §4 描述的 smem 落地。
