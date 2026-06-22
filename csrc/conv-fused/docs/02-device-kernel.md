# 02-device-kernel: device 层为什么必须像 CUTLASS

## 前言

你前面指出的最大问题是对的：如果 device 层只是一个 `run_xxx(...)` free function，
那它不像 CUTLASS。CUTLASS 的 device operator 是一个有状态 host-side object，
它持有 `Params`，暴露 `can_implement`、`initialize`、`run` 和 `operator()`。

## 官方 device 层做什么

example 13 的
`device/b2b_implicit_gemm_convolution.h` 做这些事：

- 暴露 `Arguments`，也就是 host 传入的 problem、TensorRef、epilogue params。
- `can_implement(args)` 检查 iterator、grid shape、B2B fusion shape 约束。
- `get_workspace_size(args)` 计算 split-K workspace。
- `initialize(args, workspace, stream)` 把 `Arguments` 编译成 kernel `Params`。
- `run(stream)` 计算 grid/block/smem，然后调用 `cutlass::Kernel<Kernel><<<...>>>`。
- `operator()` 串起 initialize 和 run。

所以 device 层不是“薄 launcher”。它是 CUTLASS host contract。

## tiny-cutlass 的对应关系

tiny-cutlass 不再保留本地复刻的通用 B2B device adapter。family-specific device
class 直接持有官方 example 13 的
`cutlass::conv::device::B2bImplicitGemmConvolution<Kernel>`，只负责把业务 problem
descriptor 和 raw pointer 转成 CUTLASS `Arguments`。

目标结构是：

```text
ops/conv1x1_relu_conv1x1.cu
  -> validate public API
  -> build Conv2dProblemSize
  -> device::Conv1x1ReluConv1x1<ArchTag, Element>::operator()

device/conv1x1_relu_conv1x1.cu
  -> own Arguments
  -> pack TensorRef
  -> call cutlass::conv::device::B2bImplicitGemmConvolution<Kernel>
```

这比 free function launcher 多了一点代码，但好处是边界清楚：以后替换 kernel policy
或者加 workspace/split-K/arch check，都有明确位置。

## `can_implement` 应该检查什么

在 B2B conv 里，`can_implement` 至少要覆盖：

- CUTLASS iterator 是否支持当前 problem。
- grid tiled shape 是否能被 launch encoding 表达。
- 两段 implicit GEMM 的 `M` 是否相同。
- 第一段输出 channel 是否等于第二段输入 channel。
- 第二段 conv 是否是无 halo 的 `1x1`。
- RF policy 下 `problem_N` 是否落在 threadblock tile 约束内。

指针是否为 null 更接近 runtime argument validation，可以在 `ops` 或
`initialize` 阶段拒绝。不能因为某个指针不合法就悄悄切到另一个 kernel。
