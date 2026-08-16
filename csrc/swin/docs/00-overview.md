# Swin CUTLASS 

## Overview



```text
kernel/
<TODO>

---
device/
<TODO>
---

```

约定（细节见同目录 `AGENTS.md`）：

- `DefaultPatchEmbed / DefaultSwinAttention / DefaultSwinBlock` 是 CUDA 实现侧的
  CUTLASS factory，只在 `.cu` 组装 kernel 配置，不进入 public facade header。
- 三个算子的 `can_implement` / `run` 统一返回 `cutlass::Status`；不支持的配置显式
  失败（`kErrorNotSupported`），不静默退到更弱路径。
- core runtime 用 raw device pointer + problem descriptor + `cudaStream_t`，不带
  Torch/ATen ownership。
- 当前显式实例化 `Sm80 + half_t`，构建目标 SM89。

代码分层：

```text
device/       public facade（算子声明）和内部 launch helper
kernel/       DefaultXxx factory、融合 kernel 及其 __global__ 入口
epilogue/     算子专用的 epilogue visitor（如 PatchEmbed 的 LayerNorm visitor）
threadblock/  threadblock 级计算（跨轴归约 LayerNorm、bias、gelu、residual 等）
warp/         window 坐标映射（image_token <-> window_token）
trt/          TensorRT plugin 封装
tests/swin/   host / cuDNN reference 和 executable
```

## 2. 融合原则（三个算子共用）

**GEMM/conv 只做矩阵乘；跨轴归约（LayerNorm 沿 channel、softmax 沿 key）留在 kernel
内做，不落 global 中间 buffer；所有 per-element 的 bias / 激活 / residual / shift /
partition 尽量吸附进最近的 GEMM 输入迭代器或 epilogue。** 一个完整算子内部只有它的
输入和最终输出走 DRAM，中间量留在 smem / 寄存器。

跨轴归约进不了 CUTLASS 的 `LinearCombination` 系 epilogue（它只能做 per-element
scale/bias/激活）。归约要么用 block 内 `__syncthreads` 树形归约，要么用自定义
epilogue visitor 在 warp 内 `__shfl` 归约——后者只有当归约维完整落在单个 threadblock
的 tile 内时才免跨 CTA finalize。

## 3. 各算子的目标融合形态

### 3.1 PatchEmbed = 单 megakernel（conv + bias + 完整 LayerNorm）

数学本质：stride = kernel = patch_size(4) 的 non-overlapping Conv2d patchify——把
`[B,H,W,3]` 切成 `(H/4)(W/4)` 个 `4x4x3=48` 的 patch，乘 `[embed_dim, 48]` 权重 + bias，
等价于单个 GEMM（M = B·(H/4)·(W/4) tokens，N = embed_dim = 96，K = 48）；紧跟沿
embed_dim 的 per-token LayerNorm（有偏方差，再 `*gamma + beta`）。

目标：**一个 conv kernel，epilogue 里做 bias + 完整两遍 LayerNorm**（归约 + 归一化 +
仿射），一个 kernel 出最终结果。可行支点：embed_dim=96 小，取 `ThreadblockShape::kN ≥
embed_dim` 让整个 channel 维落进单个 N-tile（`grid.n()==1`），warp 内 `__shfl` butterfly
直接出**完整** mean/var，无需跨 CTA finalize、无需第二个 kernel。

关键实现点（细节留 `01-patch-embed.md`）：

- conv fprop 本身不支持 epilogue-visitor（`ImplicitGemmConvolution` 直接调普通
  epilogue），要 fork 一个驱动 `EpilogueWithVisitor` 的 conv kernel（house style 里
  `csrc/conv-fused/conv1x1_dual` 已有 fork conv kernel 的先例）。
- LayerNorm visitor 扩展自 CUTLASS example 37 的 `EpilogueVisitorLayerNorm`（把它的
  partial 归约补成完整两遍）。
- in_channels=3 需 pad 到 8（TensorOp fp16 要 128-bit C load，禁 SIMT fallback）。

> 口径：“单 megakernel” 指**计算**融合（conv+bias+完整 LN 一个 kernel）。layout pad
> pre-pass 在把 gather/pad 折进输入迭代器之前会短期保留，不等于字面单 launch。

### 3.2 SwinBlock = 单 per-window megakernel

完整 v1 pre-norm block：norm1 + shift + partition → window attention → reverse +
residual1 → norm2 → MLP(fc1 + GELU + fc2) → residual2。

目标：**一个 threadblock 干一个 window 的全部 head**，grid = `(num_windows, batch)`，
从 partition 到 residual2 全程不落 global 中间 buffer——`normed2` / `mlp_hidden` 等
中间量留 smem，只有 block 输入和最终 output 走 DRAM。

为什么 per-window：block 里唯一的跨-token 依赖是 softmax 沿 key 归约，它把一个 window
的 L(≤64) 个 token 绑在一起；QKV / proj / LN / MLP 都作用在完整 C = heads×head_dim 上，
只有 QK^T+softmax+×V 是 per-head。让一个 block 持有完整 C 并把 head 放进 block 内循环，
attention 前后的 GEMM 和 LayerNorm 才能直接吃到完整通道，不必跨 block 拼通道；MLP
per-token 独立，顺手在同一 block 内做完，`normed2` 直接留 smem 喂 fc1。

可行域：单 kernel 的前提是一个 window 的全部中间量塞得进 smem。C 越大（Swin-T 四个
stage 的 C 从 96 涨到 768），smem 需求线性增长，深层 stage 可能超出 Sm80/89 的 ~163KB
上限；超限配置由 `can_implement` 显式 gate 拒绝，不做静默 fallback。细节留
`02-swin-block.md`。

### 3.3 SwinAttention = partition → attention → reverse

只负责 attention 子路径：window partition → window attention（QK^T + softmax + ×V +
relative-position bias）→ window reverse。attention core 复用
`csrc/flash-attention/02-tiled-online-attention` 的 flash kernel。它与 SwinBlock 只通过
内部 `WindowAttentionTensors` 共享 projection/attention 权重和 workspace，不合并成同时
含 MLP/LayerNorm 的大 tensor 结构。

## 4. 边界与非目标

- PatchMerging 是 BasicLayer stage transition，若恢复必须是独立的
  problem/factory/operator/test，不允许通过 nullable output pointer 作为 SwinAttention
  的可选副作用。
- TensorRT plugin（`trt/`）当前封装 SwinAttention，不是完整 SwinBlock 或整网。
- 本 workspace 不是完整 Swin 网络实现；BasicLayer / 整网组合待三个算子做扎实后再定。

## 5. 构建与验证

唯一入口，顺序固定 build -> verify -> bench：

```bat
scripts\kernels\swin\run.bat
```

- verify 覆盖 public 算子，失败后不 benchmark / profile。
- cuDNN correctness reference 固定在 `csrc/tests/swin/reference.h/.cpp`，不进入 runtime
  fallback；未通过 reference parity 的数据不作性能结论。
- benchmark / Nsys / NCU 采集与 CSV 解析统一在 `bench.py`。

> 稳定边界与硬约束见同目录 `AGENTS.md`；本文只描述算子边界与目标方向，实现细节随
> 各算子推进补进编号文档。
