# 01-run: 怎样运行和定位 SM80 RF example

## 前言

这一篇只解决一个问题：我要怎么跑、怎么读 `fused_two_convs_f16_sm80_rf`。
先能复现实验，再谈移植；否则我们只是在凭印象改代码。

## CUTLASS example 13 的 target

example 13 的 CMake target 不是源码文件名本身，而是加了 `13_` 前缀：

```text
13_fused_two_convs_f16_sm80_rf
```

源码文件在：

```text
3rdparty/cutlass/examples/13_two_tensor_op_fusion/fused_two_convs_f16_sm80_rf.cu
```

如果单独在 CUTLASS 顶层构建，官方路径是配置 CUTLASS，然后 build 这个 target。
在 tiny-cutlass 里，我们不把 build 输出放进 `3rdparty`，只允许走仓库的 `build/`
目录。

## 推荐阅读顺序

不要从 threadblock mainloop 直接扎进去。我的阅读顺序是：

1. `fused_two_convs_f16_sm80_rf.cu`
2. `device/b2b_implicit_gemm_convolution.h`
3. `kernel/default_b2b_conv2d_fprop_sm80.h`
4. `kernel/b2b_implicit_gemm_convolution.h`
5. `threadblock/b2b_implicit_gemm_multistage.h`
6. `threadblock/b2b_mma_base.h`

这个顺序对应 CUTLASS 的调用方向：先选 policy，再进入 device adapter，再进入 kernel
和 threadblock 算法。

## tiny-cutlass 当前验证入口

`csrc/conv-fused` 的验证不应该依赖 PyTorch extension。当前入口应保持：

```text
csrc/tests/conv-fused/
```

测试文件和 CMake target 用短名，例如：

```text
conv1x1_relu_conv1x1
conv1x1_relu_conv1x1_relu
```

Python 可以做 reference export 或验证 driver，但 C++ 运行段必须持有 raw device
pointer，并直接调用 core API。

## 当前重点

这一轮我们先把 `conv1x1 -> relu -> conv1x1` 对齐 example 13 的 RF B2B conv 风格。
也就是说，核心检查点不是“能不能跑一个结果”，而是：

- device 层是否有清楚的 `can_implement`、`initialize`、`run`、`operator()`。
- kernel 层是否只产出 `CutlassKernel`。
- RF 语义是否来自 `B2bImplicitGemmMultistage`，不是 staged global-memory 中间结果。
- 没有 SIMT/raw CUDA fallback。

`conv_relu_pool` staged baseline 已经从正式 core/build/test/plugin 路径删除。后续
恢复 pool family 时，必须先在 threadblock 层讲清楚 RF 或 SMEM accumulator 路径。
