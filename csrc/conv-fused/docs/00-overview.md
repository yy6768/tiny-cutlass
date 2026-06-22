# 00-overview: example 13 的 RF 融合卷积是什么

## 前言

这组笔记只盯一个目标：把 CUTLASS example 13 里的
`fused_two_convs_f16_sm80_rf` 读清楚，然后用同一种语言整理
`tiny-cutlass/csrc/conv-fused`。这里不是重新发明 convolution kernel，而是学习
CUTLASS 怎么把两个 implicit-GEMM convolution 放进同一个 kernel。

先说结论：example 13 的 RF 融合不是“没有 shared memory”。A/B operand 仍然照常
走 CUTLASS 的 global -> shared -> warp MMA pipeline。RF 的意思是中间激活
`D0` 不写回 global memory，也不经过 shared memory accumulator；第一段 GEMM 的
accumulator fragment 直接作为第二段 GEMM 的 A operand。

## 文件地图

example 13 的 SM80 RF 卷积路径主要经过这些文件：

- `3rdparty/cutlass/examples/13_two_tensor_op_fusion/fused_two_convs_f16_sm80_rf.cu`
  选择具体 element、layout、shape 和 `DefaultB2bConv2dFprop`。
- `3rdparty/cutlass/examples/13_two_tensor_op_fusion/device/b2b_implicit_gemm_convolution.h`
  提供 CUTLASS-style device operator：`can_implement`、`initialize`、`run`、
  `operator()`。
- `3rdparty/cutlass/examples/13_two_tensor_op_fusion/kernel/default_b2b_conv2d_fprop_sm80.h`
  选择 SM80 TensorOp implicit-GEMM 的 iterator、MMA core、threadblock MMA 和
  final epilogue。
- `3rdparty/cutlass/examples/13_two_tensor_op_fusion/kernel/b2b_implicit_gemm_convolution.h`
  定义 device kernel 的 `Arguments`、`Params`、`SharedStorage` 和 device-side
  `operator()`。
- `3rdparty/cutlass/examples/13_two_tensor_op_fusion/threadblock/b2b_implicit_gemm_multistage.h`
  真正做 RF-resident back-to-back mainloop。
- `3rdparty/cutlass/examples/13_two_tensor_op_fusion/threadblock/b2b_mma_base.h`
  提供两段 GEMM 共用的 shared storage 和 warp iterator 基类。

这套分层非常重要。`device` 层不写融合算法，`kernel` 层不启动 kernel，
`threadblock` 层不处理 host-side workspace。每层只做一件事。

## 两段卷积怎样变成 implicit GEMM

对 fprop conv 来说，CUTLASS 把卷积映射成 GEMM：

```text
A0: activation tile, logical shape M x K
B0: filter tile,     logical shape K x N
D0: accumulator,     logical shape M x N

A1: D0 after epilogue0, logical shape M x K1
B1: filter1 tile,      logical shape K1 x N1
D1: final output,      logical shape M x N1
```

其中 `M` 对应 `N * P * Q` 这类输出空间，`N/K` 对应 channel/filter 维度。
RF 融合要求两段 conv 的 `M` 一致，也就是第二段不能改变空间大小。example 13
因此要求第二段卷积通常是 `1x1` 且没有 padding halo。

## RF 融合的两个硬约束

example 13 README 里有两个约束，源码里也围绕这两个约束组织：

```text
threadblock_tile_N = problem_N
warp_tile_N        = threadblock_tile_N
```

第一个约束让一个 CTA 拥有第二段所需的完整中间 channel tile。第二个约束让一个
warp 拥有第二段所需的完整 A1 fragment。这样 `D0` 可以直接留在寄存器 fragment
里，被 `MmaTensorOpFragmentIterator` 当成 A1 operand 读走。

这就是 RF-resident 的含义：中间激活不落 global，不落 shared accumulator。

## 对 tiny-cutlass 的约束

迁移到 `csrc/conv-fused` 时，最小可接受结构应该是：

```text
ops/
  raw pointer API, problem descriptor validation
device/
  CUTLASS-style device operator, owns can_implement/initialize/run/operator()
kernel/
  DefaultXxx<ArchTag, Element, Shape...> policy factory
threadblock/
  only when we implement real RF/SMEM fusion logic
threads/
  thread-level epilogue output ops
```

如果某个路径只是顺序调用两个 CUTLASS device operator 或 tensor reduction，它最多是
staged baseline，不是 example 13 风格的融合 kernel。后续重构要把这种路径和真正的
monolithic fusion 分开，不能让它伪装成 RF fusion。

