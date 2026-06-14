# CUTLASS学习记（八）：真正的 02，IO-aware tiled online attention

## 目标

这一篇对应 `02-tiled-online-attention`，不是把 `01-online-softmax` 改个名字。

1. 面向 `sm80 / sm89 / sm90` 做 CUDA kernel 优化和编程，代码风格以 CUTLASS 2.x / 3.x 为主。
2. 这个系列的核心目的不是“写完一个 demo”，而是根据当前需求学习、验证、沉淀优化经验。
3. 每个 kernel 必须先和 reference 实现对齐，通常是 PyTorch / cuDNN / TensorRT。只有在容忍度以内，通常是 `MAE <= 1e-3`，才继续谈性能。
4. 性能结论必须来自真实 profiling 工具链：Nsight Compute、Nsight Systems，以及 `.ncu-rep`、`.nsys-rep` 和 `.csv`。
5. 新学到的东西要能被人和 AI 一起整理成中文技术博客，面向 Zhihu / X / GitHub 这类公开笔记场景。

## 为什么 `00` 和 `01` 还不是 FlashAttention

`00-naive-attention` 的数据流很直观：

```text
S = QK^T
P = softmax(S)
O = PV
```

问题也很直观：完整的 `P` 会写到 global memory。

`01-online-softmax` 只改了 softmax 这一步。它把 softmax 从普通 safe softmax 改成 online softmax，但没有改变整体 attention 的数据流：

```text
MM0 写完整 P
softmax 读写完整 P
MM1 再读完整 P
```

所以 `01` 仍然不是 IO-aware tiling。它证明了 online softmax 的数值递推是可用的，但还没有进入 FlashAttention-1 真正关心的问题：**不要把完整的 `[B, H, Sq, Sk]` 中间矩阵落到 HBM**。

真正的分界线是：

- `P` 不能作为完整矩阵存在于 global memory
- 计算要按 `Q/K/V` tile 流式推进
- tile 级别的 score / probability fragment 只能短暂存在于 register 或 shared memory
- `O` 通过 online softmax 状态逐步更新，最后归一化写回

这才是 `02` 要实现的东西。

## `02` 的数据流

`02-tiled-online-attention` 把同一个 attention 公式改写成 tile streaming：

```text
for each Q tile:
  initialize m = -inf, l = 0, O = 0
  for each K/V tile:
    MM0: S_tile = Q_tile @ K_tile^T
    online softmax update m/l on S_tile
    hand current P_tile through shared memory
    MM1: O_tile += P_tile @ V_tile
  normalize and write O_tile
```

注意这里没有完整的 `P`：

- `S_tile` 是当前 CTA 内部的 score fragment
- `P_tile` 是经过 online softmax 变换后的当前 tile fragment
- `P_tile` 通过 shared memory 交给 MM1
- global memory 里只有输入 `Q/K/V` 和输出 `O`

这就是 `02` 和 `00/01` 的本质区别。

## 本地目录

现在 tiny-cutlass 里的结构是：

```text
csrc/flash-attention/
  00-naive-attention/
  01-online-softmax/
  02-tiled-online-attention/
    CMakeLists.txt
    tiled_online_attention.cu
    kernel_forward.h
  epilogue/
  gemm/
  iterators/
  transform/
  debug_utils.h
```

`02` 来自 CUTLASS flash-attention example 的 fused forward 路径。为了让 tiny-cutlass 自己可以独立构建，我把它需要的 support headers 本地化到了 `csrc/flash-attention` 下面。

## 关键代码路径

`02` 的测试入口是：

```text
csrc/flash-attention/02-tiled-online-attention/tiled_online_attention.cu
```

真正的 fused kernel 在：

```text
csrc/flash-attention/02-tiled-online-attention/kernel_forward.h
```

核心路径可以拆成五步：

1. `MM0` 用 CUTLASS MMA 计算 `Q_tile @ K_tile^T`
2. `iterative_softmax(...)` 在当前 score tile 上更新 `m / l / out_rescale`
3. `MM0::B2bGemm::accumToSmem(...)` 把当前 attention tile 写入 shared memory
4. `MM1` 从 shared memory 读取 attention tile，再和 `V_tile` 做 MMA
5. epilogue 负责按 online softmax 的分母归一化并写回 `O`

这不是“一个更快的 softmax kernel”，而是把 attention 的中间数据生命周期从 HBM 移到片上。

## 为什么这是 IO-aware

单看 `MM0` 和 `MM1`，它们仍然是 tensor core GEMM 路径；它们的局部指标甚至可能看起来很像 `00/01` 里的两个 GEMM。

但 FlashAttention 的动机不是让某一个局部 kernel 看起来更轻，而是减少整个 attention 的 HBM 读写：

```text
00/01:
  Q/K/V -> MM0 -> full P in HBM -> softmax -> full P in HBM -> MM1 -> O

02:
  Q/K/V tiles -> MM0 -> P_tile on chip -> MM1 -> O
```

所以判断 `02` 是否方向正确，应该优先看：

- 是否完全避免 full `P` materialization
- 中间状态是否限制在 tile scope
- HBM traffic 是否减少
- reference parity 是否通过

而不是只看某个局部 kernel 的 tensor core utilization。

## 构建和验证顺序

这个系列的固定顺序是：

```text
build -> verify -> bench
```

`02` 必须先编译通过，再跑 reference check。只有 correctness 通过以后，benchmark 和 Nsight 报告才有意义。

当前 CMake 目标是：

```text
flash_attention_02_tiled_online_attention_test
```

聚合目标 `flash_attention` 也已经包含 `00 / 01 / 02`。

## 性能分析产物

后续 profiling 应该把产物放在 `build/` 下，并同时保留：

- `.ncu-rep`：给人类看
- `.csv`：给 agent 分析
- `.nsys-rep`：看系统级时间线

性能结论必须绑定 reference parity。没有 correctness 的性能数字，不应该进入博客结论。

## 下一步

`02` 完成以后，后续变体才应该继续讨论：

- tile shape
- shared memory handoff
- warp 分工
- work partitioning
- FA2 风格的并行组织

这些优化都建立在一个前提上：完整 `P` 已经从 global-memory 路径里拿掉了。
