# Swin workspace 记忆

这个文件只记录会影响后续修改的稳定边界和已知约束。
`window_attention` 现按用户提供的 attention 和后续要求重建：必须融合 RMSNorm、使用 reflect、提前展开位置偏置；
其他 family 仍保留标准 Swin stage 的 LayerNorm 语义。

## 设计线

- 当前是 **window=4（L=16）简化线**，全 CUTLASS **2.x**。
- 选 2.x 不是偏好：CUTLASS 4.5.2 的 `collective_builder.hpp` 只特化
  Sm90/100/103/120，**没有 Sm80/89 特化**。3.x 在本机 SM89 上等于手搓
  `CollectiveMma<MainloopSm80CpAsync, ...>`（唯一先例 examples/59），且 conv
  implicit-GEMM、epilogue visitor、from-smem B2B 在 3.x Sm80 上都无现成件。
  2.x 的 `kernel/threadblock/warp/epilogue` 目录惯例也正是本仓库要求的形态。
- 不新增 3.x/CuTe 变体，除非用户明确要求。

## 四个 family（不是五个）

| family | 内容 |
|---|---|
| `patch_embed` | conv(patch×patch, stride=patch) + bias + 沿 channel 完整 LN |
| `window_attention` | reflect/RMSNorm → grouped QKV → softmax((QKᵀ+bias)·scale)·V → proj+bias → reverse/crop，单 kernel |
| `swin_mlp` | LN2 → fc1 + GELU → fc2 + residual2 |
| `patch_merging` | 2×2 空间聚合 → LN → 线性 4C→2C |

## window partition / reverse 不是独立 kernel

已核实的机制（**不要**为此写 bespoke iterator）：

- `GemmUniversal` 自带 `GatherA`/`GatherB`/`ScatterD` 模板 flag
  （`gemm_universal.h:130-136`），范例 `examples/36_gather_scatter_fusion`。
- gather 语义是**按行号重映射**：`predicated_tile_access_iterator.h:575`
  的 `if (Gather) coord_strided = indices_[coord_strided];`；A row-major 时
  strided rank 就是 GEMM M。
- `ScatterD` 在 `epilogue/threadblock/predicated_tile_iterator.h:336,410`
  的 load 和 store 两条路径上都生效 ⇒ residual1 可以用**同一张索引表**
  经 source-C 读回。
- 一个 token 的 C 个 channel 连续 ⇒ 一个 token 就是一行 ⇒ partition 折进 QKV
  GEMM 的 `GatherA`，reverse 折进 proj GEMM 的 `ScatterD`。cyclic shift 烘进表。

所有表在 host 侧构建，见 `window_index.h`。**改动那里的坐标约定必须重跑
`swin_window_index`。**

上述 `GemmUniversal` 是已有机制测试。当前 fused attention 使用 CUTLASS
`DefaultMmaTensorOp` 的 shared-memory iterator，QKV 输入加载和 proj 写回
直接消费 `window_attention/window_index.h` 的 host 行索引，没有独立 partition
kernel，也没有新增 bespoke iterator。reflect 的 gather/scatter 必须分开：
halo 只读不写（scatter=-1），防止重复输出写入。改动该表须重跑 attention
的 PyTorch 全套对拍和 graph replay。

## Descriptor

- `swin_problem.h`：`PatchEmbedProblem` / `SwinStageProblem` / `PatchMergingProblem`，
  纯 scalar，无 device pointer、无 CUTLASS 类型、不依赖 Torch/ATen。
- `SwinStageProblem` 的 `window_size`/`shift_size` 是**请求值**；一律经
  `effective_window()` / `effective_shift()` 读取（官方 clamp：window ≥ 特征图
  时退化成全局 attention 且 shift=0）。
- core runtime 入口收敛到 raw device pointer + problem descriptor + `cudaStream_t`。
- `window_attention/problem.h` 的 `WindowAttentionProblem` 专用于用户的 DLSS4
  参考：window 固定 4，shift_h/shift_w 独立且不做官方小图 clamp。
  QK 可 shared/separate，QK/value 维度彼此独立。position bias 必须提前展开
  为二维 `[G*16,16]`，不在 kernel 内计算 relative index。边界固定 reflect，
  支持重复反射，小图单像素轴按复制处理；不做 cyclic shift 或 attention mask。
  未给出的 helper 约定见 `window_attention/README.md`。
- attention 当前是 FP16 输入、FP32 MMA/softmax 累加的推理路径；必须融合输入
  RMSNorm（gamma 必需，默认 epsilon 为 FP16 finfo epsilon），不含残差或 MLP。FP8 和非零 dropout 显式拒绝，不允许默认切换语义。
- 完整 DLSS4 block 使用同一 family 的 `ops/swin_block.h`，包含两个独立 RMSNorm、
  attention、两次残差及 32→128→32 的 erf-GELU MLP，单 kernel。用户指定的主性能
  用例为 `[1,720,1280,32]`、window4、ratio4、G1/Dq32/Dv32，不能用小图性能代替。
  `swin.bat block` 执行 build→完整 block verify→Graph bench。core buffer 要求
  16-byte 对齐，无 workspace；其他 C/ratio/head shape 显式 unsupported。
  复用 attention MMA 的标量访存模板是旧 attention API 路径；完整 block 选择
  128-bit 访存模板，两条路径均须保持 reference parity。
- attention 必须保留 `(QKᵀ + bias) * scale` 顺序，以及 eager FP16
  在 QKV、scores、bias 加法、scale、softmax 和 PV 输出处的舍入。
- `window_attention/kernel/default_grouped_gemm.h` 是单独验证的 grouped GEMM
  primitive；SM89、CUDA≥12.4、FP8 同格式 A/B、FP32 输出，指令为
  `mma.sync.aligned.m16n8k32`。它不表示完整 attention 已支持 `quant_fp8=True`。
  `swin.bat grouped-gemm` 跑该验证，PTX/SASS 和 reports 放 `build/swin/`。

## house style（继承全局 AGENTS.md + conv-fused 惯例）

- `ops/` 暴露 API，`device/` 做 CUTLASS device operator 与显式支持性检查，
  `kernel/` 只装配 `CutlassKernel`。
- primary 保持 `DefaultXxx<ArchTag, Element, ThreadblockShape, WarpShape>` 工厂风格；
  dtype/arch/layout 不进 primary 名字。
- 不支持的配置返回 `kErrorNotSupported`，**无 SIMT / raw CUDA fallback**。
- 不要引入 `Impl_` 式双入口，也不要为保留 staged golden 加 public operator 变体。
- 只建有真实内容的目录，不留空壳；target 不加 `_test` 后缀。

## 已确认的技术约束

- **conv fprop 不支持 epilogue visitor**：`ImplicitGemmConvolution::operator()`
  直接调普通 threadblock epilogue。visitor（`begin_row`/`end_row`，
  `epilogue_with_visitor.h:117,134`）只在 GEMM kernel 层。所以 patch_embed 的
  LN 需要 fork 一个 conv kernel 去驱动 `EpilogueWithVisitor`；先例见
  `csrc/conv-fused/conv1x1_dual/kernel/b2b_implicit_gemm_convolution.h`。
- **单 kernel 完整 LN 的支点**：取 `ThreadblockShape::kN >= embed_dim` ⇒ 整个
  channel 维落进单个 N-tile ⇒ `end_row` 里 `__shfl_xor_sync` butterfly 直接出
  完整 mean/var，无需跨 CTA finalize。范式 `examples/37`。
- PatchEmbed `stride == kernel == patch` ⇒ 零 halo、im2col 放大 1.0；但 NHWC 下
  一个 patch 是 4 段不连续内存，**不能**退化成 plain GEMM + GatherA，必须走 conv。
- activation channel 3→8 pad 是必需的（TensorOp fp16 要 8 宽 C load，禁 SIMT
  fallback）；v1 保留一个廉价 pad pre-pass，口径写成"计算单 kernel + 1 个 pad pre-pass"。

## 易错点（改代码前先读）

- `LN_pe` / `LN1` / `LN2` 是**三个不同 norm，不能合并** —— residual 锚点在它们之间取值。
- im2col 的 `(c,ky,kx)` 展开顺序必须严格匹配 host 侧 `Wpe.reshape(C,48).T`。
- proj 的 `output_bias` 必须在 reverse/residual1 **之前**加。
- MLP hidden 切块后 `b2` + residual2 只能在循环**外**加一次。
- LN 方差用**有偏**（除 N），fp32 累加。
- `GatherA` 索引是**行号**，不是字节偏移。
- SW-MSA mask 用 `-1e4` 不用 `-inf`（fp16 存储时 -inf 会让 softmax 的减最大值出 NaN）。
- GELU 必须 erf 精确版（tanh 近似有 ~1e-3 系统偏差，会被误判成 bug）。
- PatchMerging 的 norm 在 GEMM **之前**，与 PatchEmbed 相反。

## 验证与文档

- 统一入口 `scripts/kernels/swin/swin.bat`（`run.bat` 转发），顺序固定 build → verify → bench；
  verify 失败不得 benchmark/profile。
- harness 在 `csrc/tests/swin/{verify.py,bench.py,window_attention.cu}`。
  attention 要求 MAE ≤ 1e-3 且 max_abs ≤ 2e-2，非有限值立即 fail；
  fixture、benchmark 和 NCU/NSYS 报告均只放 `build/swin/`。
- 判据 MAE ≤ 1e-2（fp16 输入、fp32 累加），同时打印 `max_abs`；非有限值立即 fail。
  seed 固定 2026。
- 复用 `csrc/tests/common/test_utils.h`（`DeviceBuffer`、`fill_random_uniform`、
  `compare_host`）和 `cudnn_conv_reference.h`。cuDNN reference **必须是独立 host TU**
  （`cudnn_frontend.h` 进 `.cu` 会撑爆 cudafe++），见
  `csrc/conv-fused/CMakeLists.txt`。cuDNN 不进 runtime fallback。
- `CMakeLists.txt` 里只注册**已通过对拍**的 target，`--target swin` 永远可验证。
- 学习和设计文档放 `docs/`，按 `00-...` 连续编号；删除过期路径，不保留与当前实现
  冲突的历史说明。

## 当前进度

- **step 0 完成**：`swin_problem.h`、`window_index.h`、`swin_window_index` gate
  全绿（partition/reverse 往返、gather 索引是排列、mask 对称性、rel-pos 镜像、
  patch-merging 覆盖、stage4 clamp）。
- 后续步骤见 `docs/00-overview.md`。
- DLSS4 attention 已完成 RMSNorm + 单 kernel TensorOp 流程，66 个 PyTorch 用例和
  CUDA Graph 单节点 replay 已通过。现有 overview 中标准 Swin v1 的
  attention/LN1/residual1 设计不作为该 API 的语义来源；以用户参考及
  `window_attention/README.md` 为准。
