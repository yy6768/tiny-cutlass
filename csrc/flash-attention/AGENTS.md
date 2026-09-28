# FlashAttention 记忆

## 目标

这个目录记录 tiny-cutlass 当前 FlashAttention 学习路线。

- 围绕 SM80、SM89、SM90 做优化；CUTLASS 2.x 或 CUTLASS 3.x 风格按实验需要选择。
- 每个 kernel family 都是一轮学习闭环：理解算子、实现 kernel、测量、解释变化，
  再把经验保留下来。
- 性能数据可信之前，必须先对齐 PyTorch、cuDNN、TensorRT 或其他明确 reference。
- 只有数值误差进入要求范围后，才能把 kernel 当作正确实现；默认目标是
  MAE <= 1e-3，除非具体实验另有阈值。
- Nsight Compute 和 Nsight Systems 是主要性能工作流；`.ncu-rep`、`.nsys-rep`
  和 CSV 都是一等 profiling artifact。
- 学到的内容要沉淀成中文技术笔记，适合后续发布到知乎、X、GitHub 等渠道。
- fallback implementation 尽量少；依赖缺失或配置不支持时，优先用 CMake 检查或
  显式 unsupported path，而不是在 kernel/test harness 里藏 fallback。
- 当前优化重点是 SM80 FP16 和 SM89 FP8，除非用户明确扩大架构或 dtype 范围。

## 工作流

1. build kernel family。
2. 用选定 reference 做 verify。
3. verify 通过后才 benchmark。
4. 需要性能证据时，用 Nsight 做 profile。

## 约定

- kernel step 按 `00`、`01`、`02` 这样的顺序编号。
- `00` 通常是 family baseline，但目录不锁死固定 step 数。
- 编号 kernel 的 launch TU 保持为 launch entry。`csrc/tests/flash-attention` 下的共享
  executable 负责输入生成、reference 执行、MAE 检查和计时。
- `02-split-kv` 起改用 ex13 风格分层布局（device/kernel/threadblock/warp/epilogue），launch TU
  收在 `device/flash_attn.cu`，launch 逻辑由 device 层的 `run()` 经
  `cutlass::Kernel<T>` trampoline 发出；不再保留 step 根目录的离散 `.cu` 和
  umbrella header。
- kernel 源文件允许中文注释（学习笔记风格）；GBK locale 下 EDG 前端会把 UTF-8
  中文注释误解析成声明延续，必须带 `/utf-8` 编译（`flash_attention_kernels` 的
  CMake 选项里已加，新建 target 时保持）。
- fixed-seqlen flash-attention 测试当前使用 cuDNN SDPA 作为 reference backend。
  cuDNN 依赖必须在 CMake configure 阶段检查并提前失败；不要新增
  `HAS_CUDNN` 风格的 C++ fallback 分支。
- 所有编号 kernel variant 都要注册到 `flash_attention.h` 的共享 `Kernel` 接口；
  `flash_attention_test --kernel=all` 必须覆盖 00/01/02 和后续迁移变体。
- C++ 结构偏向 CUTLASS example 风格：公开学习接口保持简单全局作用域，文件局部
  helper 使用匿名 namespace。这个工作区不要新增
  `tiny_cutlass::flash_attention` 这类多层项目 namespace。
- `blogs/` 只放笔记。
- 构建和 profiling 产物都放在 `build/`。
- 除非用户明确扩大范围，修改保持在当前 family 内。

## 当前 kernel

- `00-naive-attention` 会 materialize 完整 `P` 矩阵，是 baseline。
- `01-online-softmax` 只改变 softmax kernel，仍然会 materialize `P`。
- `02-split-kv` 是重置后 LeetCUDA MMA 路线的第一个 kernel：FlashAttention-1
  forward，Q/K/V 全部按 warp 切分（split-KV warp tiling），m16n8k16 MMA 走
  CUTLASS 包装。旧 `02-tiled-online-attention`（example 41 风格 fused kernel）
  已废弃并从构建移除，目录仅作历史保留，不要再接入构建或测试。
- `02-split-kv` 使用标准 canonical TensorOp iterator，四个 warp 沿 KV/输出通道切分，
  每个 warp 覆盖全部 Br 行。device 按48KiB容量公式选择 Br/Bc；FP16输入输出、
  FP32累加，D=Dv 为32/64/96/128，D128 的逻辑 Bc=48、物理 KV 宽度=64。
  Q/K 使用 `cp_async`，V 使用 global load 和转置 ThreadMap；K/V 单 buffer，Q/P union复用。
  外层KV、内层Q：一个CTA负责一个batch/head（官方FA1 num_splits=1结构），
  K/V每个外层迭代只搬一次，供全部Q tiles复用。完整双层循环在threadblock内。
- `02-split-kv` 的 softmax 和输出更新按 FA1 Algorithm 1 第10～12行：
  tilde P=exp(S-tilde m)，l_new=alpha*l_old+beta*tilde l，
  每轮 O_new=(alpha*l_old*O_old+beta*tilde P V)/l_new。
  不提前归一化 P，也不把输出归一化推迟到最终 epilogue；epilogue 只转换和写回。
  中间归一化O和显式m/l使用FP32 workspace，首轮直接初始化寄存器，不读旧workspace；
  后续轮读取上一轮状态，最终轮仅将O转换到FP16输出。Q行padding到Br倍数，
  workspace大小为B*H*ceil(Sq/Br)*Br*(Dv+2)*sizeof(float)，必须检查容量与16字节对齐。
  不分配全局S/P，不用跨CTA原子合并；Q/KV按完整tile在前、尾tile在后推进。
  `MmaTensorOpAccumulatorTileIterator`负责中间O读写；m/l行映射是attention自定义逻辑。
  官方FA1使用LSE，本实现暂保留论文显式m/l；不声称逐项照搬官方所有优化。
  博客在仓库根目录 `blogs/flash-attn/02-split-kv.md`；固定Q版本的NCU不能用作新循环证据。

## 03-split-q

- `03-split-q/device/flash_attn.cu` 注册 forward 到共享 `Kernel` 接口，核心仍为
  raw pointers + Problem + cudaStream_t。`Tensors::logsumexp` 是可选训练输出。
- `device/flash_attn_backward.cu` 提供独立 `BackwardKernel` 接口，CTA沿KV分块、
  扫描Q，用5个CUTLASS TensorOp计算梯度。dK/dV由CTA独占，dQ经FP32 workspace原子累加。
  当前仍在严格梯度parity候选验证中；不得把历史C03 forward数据称为backward结果。
- backward的dS shared使用RowMajor congruous写入、ColumnMajor congruous读取转置；
  dQ warp必须M32、N>=32，按2×2划分输出，不得改回不支持的M16 congruous A偏移。
  每次run在同一stream清空dQ workspace，device显式检查并申请dynamic shared opt-in。
- `DefaultFlashAttnSplitQ` 选择 Br=Bc=64，4 个 warp 分别拥有16行 Q 和完整输出通道；
  m/l/O 保留寄存器，Q 只搬一次，P 通过 example 13 的
  `MmaTensorOpFragmentIterator` 传给 PV，不写 shared。
- Q/K 使用标准 canonical TensorOp iterator；V 使用 `DefaultMmaCore` 的
  `SmemLayoutB`、`IteratorThreadMapB`、`SmemIteratorB` 和128-bit `cp_async`。
  MmaCore 的 stage=3 仅选择 SM80 access-iterator 类型；实际只有一个 V buffer。
- D=Dv 为32/64/96/128，D96 的 V 物理通道补齐128；合法输入/输出仍是96。
  不要去掉 global load 与 epilogue 的通道边界 predication。
- KV 必须按完整块在前、尾块在后的次序推进。C01 的 residue-first 次序在
  FP16 P 舍入后不能满足严格 MAE；详见 `docs/03-split-q-results.md`。
- 入口 `scripts/kernels/attention/attention.bat`，顺序 build→verify→bench；
  `verify.py --kernel=03-split-q` 强制 MAE<1e-6，`bench.py` 检查同一二进制的
  parity stamp，再对四个既有 shape 分别检查 t03/t02<0.95。
- 同一入口在forward verify之后运行 `verify.py --phase=backward`；两个阶段均通过才计时。
  backward端到端验证使用候选/cuDNN各自的O/LSE，逐个梯度MAE<1e-6；
  显式diagnostic-state只用于误差归因，不产生验收stamp或计时。
- 已在 RTX 4070 Laptop SM89 测试；SM80 policy 已实例化，但没有 SM80 硬件实测。

## 共享测试入口

- `flash_attention_test` 是所有已注册 kernel 的共享测试 executable。
- 可用参数包括 `--kernel=list`、`--kernel=00-naive`、
  `--kernel=01-online-softmax`、`--kernel=02-split-kv`、`--kernel=03-split-q`、`--kernel=all`。
- 每个 kernel 另有独立 executable（`naive_attention`、`online_softmax_attention`、
  `split_kv_attention`、`split_q_attention`），指向同一个共享 host C++ test main，只是默认 kernel 不同；
  脚本入口为 `scripts/kernels/attention/02-split-kv-attention.bat` 等同名 `.bat`。
