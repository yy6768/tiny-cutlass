# Swin Transformer on CUTLASS 2.x — Overview

这一系列文档记录从 0 重建 Swin 管线的设计。约定见同目录 `AGENTS.md`。

## 1. 为什么是 2.x

CUTLASS 4.5.2 的 `include/cutlass/gemm/collective/collective_builder.hpp` 只特化
Sm90 / Sm100 / Sm103 / Sm120，**没有 Sm80/89 特化**。本机是 SM89（RTX 4070 Laptop），
所以 3.x/CuTe 在这里意味着手搓 `CollectiveMma<MainloopSm80CpAsync, ...>`——全 CUTLASS
仓库只有 `examples/59_ampere_gather_scatter_conv` 一个先例，而且 conv implicit-GEMM、
epilogue visitor、from-smem B2B 在 3.x Sm80 路径上都没有现成件。

2.x 的分层惯例（`kernel/` → `threadblock/` → `warp/` → `epilogue/`）本身就是这个仓库
要求的目录形态，两者刚好对齐。

## 2. 管线分解

一个完整的 v1 pre-norm Swin block 加上两端的 stage transition：

```
PatchEmbed:      conv(4x4, stride 4) + bias + LayerNorm(channel)
                 |
                 v
  ┌──────────── SwinBlock ────────────┐
  │  x0 = x                            │  <- shortcut1
  │  LN1                               │
  │  window partition (含 cyclic shift)│  <- 折进 QKV GEMM 的 GatherA
  │  QKV projection                    │
  │  softmax(scale·QKᵀ + bias + mask)  │
  │  · V                               │
  │  proj (C→C) + bias                 │
  │  window reverse                    │  <- 折进 proj GEMM 的 ScatterD
  │  x1 = x0 + ...                     │  <- residual1，走 ScatterD 的 source-C
  │  LN2                               │
  │  fc1 (C→4C) + GELU                 │
  │  fc2 (4C→C)                        │
  │  x2 = x1 + ...                     │  <- residual2
  └────────────────────────────────────┘
                 |
                 v
PatchMerging:    2x2 空间聚合 → LayerNorm(4C) → 线性 4C→2C
```

Swin 的 block 总是成对出现：第一个 shift=0（W-MSA），第二个 shift=w/2（SW-MSA）。

## 3. 关键设计决策

### 3.1 window partition / reverse 不是 kernel

这两步是纯数据搬运：memory bound，白跑两趟 HBM，没有任何算术可以把延迟藏进去。
CUTLASS 2.x 已经有正好合适的钩子——`GemmUniversal` 自带 `GatherA` / `GatherB` /
`ScatterD` 三个模板 flag（`gemm_universal.h:130-136`），gather 是**按行号重映射**：

```cpp
// predicated_tile_access_iterator.h:575
if (Gather) coord_strided = indices_[coord_strided];
```

A 是 row-major 时 strided rank 就是 GEMM 的 M 维。一个 Swin token 的 C 个 channel 在
内存里连续 ⇒ **一个 token 就是一行** ⇒ 一行一个索引：

- partition ⇒ QKV GEMM 用 `GatherA`，`idx[m] → token_id`
- reverse ⇒ proj GEMM 用 `ScatterD`，同一张表
- residual1 ⇒ 走 `ScatterD` 的 source-C 路径（`predicated_tile_iterator.h:336,410`
  的 load 和 store 两条路径都被同一索引重定向）

cyclic shift 烘进索引表，kernel 侧零索引数学。**零 bespoke iterator 代码。**

已在 device 上逐 bit 验证（`swin_window_gather_scatter`，max_abs 恰好 0.0）。

一个容易踩的点：predication 用的是**逻辑 extent**（problem_size 的 M），而 gather 改的是
**地址**。所以 A buffer 的行数可以和 M 不同——测试里 A 有 `num_tokens` 行而 M 是
`num_window_rows`。这与 example 36 用 `GatherB` 的方式一致。

### 3.2 window=4 让 attention 大幅退化

`L = w² = 16` 正好是一个 `m16` MMA tile：

- 无 padding（对比 window=7 的 L=49 要 pad 到 56）
- `S[16,16]` 全驻寄存器
- softmax 退化成 lane 内 4 元素归约 + 2 次 `__shfl_xor_sync` 单趟完成，
  **不需要 flash 的 online rescale 状态机**
- rel-pos-bias 与 shifted mask 可在 host 侧折成**一张相加表** `[class, heads, 16, 16]`，
  window=4 / heads=3 时只有 12 KB，L2 常驻

### 3.3 shifted mask 只有 4 个类

只有贴着卷绕边缘的 window 才带 mask，所以 mask 只取决于 window 的
(是否最后一行, 是否最后一列)——最多 4 类，shift=0 时退化成 1 类。窗口内每个 slot 按
其**卷绕后**坐标落在哪个区间打标签，标签相同才允许互相 attend。这和官方 Swin 对
未卷绕的 `img_mask` 做 partition 是等价的。

mask 值取 `-1e4` 而不是 `-inf`：scores 用 fp32 累加但可能以 fp16 存储，`-inf` 会让
softmax 的减最大值步骤在整行被 mask 时产出 NaN。

### 3.4 PatchEmbed 的 LayerNorm 需要 epilogue visitor

- conv fprop **不支持** visitor：`ImplicitGemmConvolution::operator()` 直接调用普通的
  threadblock epilogue。visitor 机制（`begin_row` / `end_row`，
  `epilogue_with_visitor.h:117,134`）只存在于 GEMM kernel 层。
- 普通 `EpilogueOutputOp` 是逐元素 thread functor：没有跨 lane 钩子、没有两遍，
  结构上做不了沿 channel 的归约。
- **单 kernel 完整 LN 的支点**：取 `ThreadblockShape::kN >= embed_dim` ⇒ 整个 channel
  维落进单个 N-tile ⇒ `end_row` 里 `__shfl_xor_sync` butterfly 直接出**完整**
  mean/var，无需跨 CTA finalize。范式见 `examples/37`。
- `stride == kernel == patch` ⇒ 零 halo、im2col 放大倍数 1.0。但 NHWC 下一个 patch 是
  4 段不连续内存，所以**不能**退化成 plain GEMM + GatherA，必须走 conv。

## 4. 问题规模（window=4 设计线）

```
B=1, image=224, patch=4, embed_dim=96, window=4, mlp_ratio=4
stage1: 56x56x96,  heads=3,  head_dim=32, hidden=384
stage2: 28x28x192, heads=6,  head_dim=32, hidden=768
stage3: 14x14x384, heads=12, head_dim=32, hidden=1536
stage4:  7x7x768,  heads=24, head_dim=32     <- 4 除不尽 7
L = 16 (无 padding), shift = 2, rel-pos table = (2*4-1)^2 = 49 / head
```

stage4 上 `window=4` 除不尽 `7`。按官方规则处理（window ≥ 特征图时退化成全局 attention
且 shift=0），写进 `can_implement`；本轮验证覆盖 stage1–3。

## 5. 文件结构

```text
csrc/swin/
  swin_problem.h        PatchEmbedProblem / SwinStageProblem / PatchMergingProblem
  window_index.h        host 侧索引表、bias+mask 折叠表、朴素 partition/reverse 参考
  docs/                 00 overview, 01.. 逐个 family

  <family>/
    ops/                公共 API：raw device pointer + problem + cudaStream_t
    device/             CUTLASS device operator + 显式 can_implement
    kernel/             DefaultXxx<ArchTag, Element, TBShape, WarpShape> 工厂
    threadblock/        CTA 级 MMA 组合
    warp/               warp 级原语（如 window softmax）
    epilogue/ threads/  output iterator / output op

csrc/tests/swin/        每个 family 一个自检 target，打印 MAE / max_abs
```

## 6. 进度

| step | 内容 | 状态 |
|---|---|---|
| 0 | descriptor + 索引/bias/mask 表 + host gate | ✅ `swin_window_index` PASS |
| 2 | GatherA / ScatterD == partition / reverse | ✅ `swin_window_gather_scatter` PASS，max_abs 0.0 |
| 1 | `patch_embed` | 进行中 |
| 3 | `window_attention` | 待做 |
| 4 | `swin_mlp` | 待做 |
| 5 | `patch_merging` | 待做 |
| 6 | stage 端到端 | 待做 |

step 2 提前到 step 1 之前跑，因为它验证的是整个设计的核心假设——先证伪比先动工便宜。

融合（attention+MLP 同 kernel、PatchEmbed 作 prologue）和 FP8 都不在本轮范围。
