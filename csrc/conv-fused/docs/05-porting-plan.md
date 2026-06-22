# 05-porting-plan: tiny-cutlass 的重构路线

## 当前结论

`conv1x1 -> relu -> conv1x1` 可以直接对齐 example 13 的 RF B2B conv 风格。
它天然满足第二段 `1x1`、无 halo、空间 `M` 不变的条件。

`conv3x3 -> relu -> pool -> conv1x1 -> relu -> pool` 不能直接套
`DefaultB2bConv2dFprop`。pool 会改变空间 `M`，并且 maxpool 是局部 reduction，
不是第二个 implicit-GEMM。它需要独立 threadblock 设计。

## 第一阶段：清理 device contract

必须完成：

- `device/conv1x1_relu_conv1x1.{h,cu}` 改成 CUTLASS-style class。
- 保留 `ops/` raw pointer API，但它只负责公共参数验证和 problem descriptor。
- 移除 device 层的 standalone `run_*` launcher 作为主要入口。
- 不加旧名 alias，不做兼容 wrapper。

## 第二阶段：隔离 staged pool baseline

被删除前，`conv_relu_pool` 是 staged CUTLASS baseline：

```text
conv device op -> pool reduction device op -> conv device op -> pool reduction device op
```

它不是 single kernel，也不是 RF fusion。后续有两种干净选择：

- 删除 core build/test/plugin 中的 staged path，只留下设计文档和 TODO。
- 或者把它移到明确的 `staged/` family，名称、文档、target 都不能暗示它是 fused RF
  kernel。

在用户要求“完全重构、不留 fallback”的约束下，这条 staged 正式路径已经删除。后续
如果要恢复 pool family，应该重新建立真实 threadblock-level RF/SMEM 设计，而不是
把 staged baseline 改名搬回来。

## 第三阶段：设计 pool 的 threadblock 语义

真正的 conv+pool fusion 需要回答：

- pool window 是否只消费同一个 CTA 内产生的 output fragment。
- 2x2 stride2 reduction 是否可以在 RF fragment 层完成。
- 如果跨 warp 或跨 CTA，是否要显式 SMEM accumulator。
- pool 后的 conv1x1 如何获得连续的 A1 fragment。

只要这些问题没有在 threadblock 层被回答，就不能写 device wrapper 假装已经完成
monolithic fusion。

## 第四阶段：验证和 benchmark

验证顺序保持：

```text
build -> verify -> bench
```

没有 reference parity 的 kernel 不做性能结论。C++ 执行段使用 raw pointer；
PyTorch、TensorRT、ModelOpt 只作为外部 reference 或 driver。
