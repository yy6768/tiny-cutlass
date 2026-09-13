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
- `02-split-kv` 当前使用 CUTLASS `cp_async` 组件，device policy 选择2-stage K 环形缓冲；
  Q/V 保持单 buffer，FP16 输入输出与 FP32 累加，显式支持 D=Dv 为32/64/96/128。
  QK 使用 2×4 warp 切分，PV 每 warp N=32、内部通道补齐到128；
  不要直接将 congruous B iterator 的 warp N 改成8/16，必须验证其地址置换。
  完整边界、验证和 NCU 证据见 `blogs/02-split-kv.md`。

## 当前测试入口

- `flash_attention_test` 是所有已注册 kernel 的共享测试 executable。
- 可用参数包括 `--kernel=list`、`--kernel=00-naive`、
  `--kernel=01-online-softmax`、`--kernel=02-split-kv`、`--kernel=all`。
- 每个 kernel 另有独立 executable（`naive_attention`、`online_softmax_attention`、
  `split_kv_attention`），指向同一个共享 host C++ test main，只是默认 kernel 不同；
  脚本入口为 `scripts/kernels/attention/02-split-kv-attention.bat` 等同名 `.bat`。
