# 04-rf-threadblock: RF-resident mainloop 的核心

## 前言

这一篇只读 threadblock 层。device 和 kernel 层只是把类型准备好，真正把两个 tensor
op 接起来的是 `B2bImplicitGemmMultistage`。

## 第一段 mainloop

第一段 implicit-GEMM 做普通 TensorOp pipeline：

```text
global A0/B0 -> shared A0/B0 -> warp fragments -> mma -> accum0
```

这里 shared memory 只用于 A0/B0 operand staging。`accum0` 是 warp-level accumulator
fragment，仍然在寄存器里。

## 中间 epilogue 不写 global

第一段完成后，RF 路径没有调用一个 threadblock epilogue 把 `D0` 写出。它做的是：

```text
accum0
  -> FragmentIteratorA1
  -> output_op_0(relu / scale / bias)
  -> A1 warp fragment
```

这一步是 example 13 最值得学的地方：epilogue0 是被“顺手”应用在 A1 fragment
加载过程里的，而不是作为一个单独 kernel 或 global store。

## 第二段 mainloop

第二段的 B1 filter 仍然需要 global -> shared -> warp fragment pipeline：

```text
global B1 -> shared B1 -> warp fragment B1
```

A1 则来自 `accum0`。随后第二段 TensorOp 产生 `accum`，最后由普通 CUTLASS epilogue
写出 `D1`。

## RF 与 SMEM accumulator 的分界

RF 版本：

```text
accum0(register)
  -> FragmentIteratorA1
  -> mma1
```

SMEM accumulator 版本：

```text
accum0(register)
  -> EpilogueSmemAccumulator
  -> shared accumulator
  -> WarpIteratorA1
  -> mma1
```

所以以后如果我们实现 pool 融合，必须先回答一个问题：pool 的 reduction 能不能在
同一个 warp/CTA 的 register fragments 内完成。如果不能，就应该显式设计 SMEM 路径，
而不是把 staged global-memory baseline 叫做 RF fusion。

