# 03-kernel-policy: DefaultB2bConv2dFprop 怎样装配 RF 路径

## 前言

`DefaultB2bConv2dFprop` 是 example 13 的核心类型工厂。它不是运行时对象，而是一个
compile-time policy assembler：给它 element、layout、arch、threadblock shape、
warp shape、epilogue op，它产出一个 `Kernel` type。

## SM80 RF 路线的关键类型

在 `kernel/default_b2b_conv2d_fprop_sm80.h` 里，RF 路径选择的是：

```text
DefaultMmaCore<...> for op0
DefaultMmaCore<...> for op1
Conv2dFpropActivationTileAccessIterator*
Conv2dFpropFilterTileAccessIterator*
MmaTensorOpFragmentIterator for A1
B2bImplicitGemmMultistage
DefaultEpilogueTensorOp for final output
```

真正说明 RF 融合的是 `MmaTensorOpFragmentIterator`。它不是从 global memory 加载
A1，也不是从 shared accumulator 加载 A1，而是从第一段 accumulator fragment 里把
A1 warp tile 读出来。

## 为什么 kernel 层不应该有 launch 逻辑

kernel 层只应该定义：

```cpp
using CutlassKernel = typename cutlass::conv::kernel::DefaultB2bConv2dFprop<...>::Kernel;
```

它不应该：

- include `cutlass/conv/device/*`
- 创建 device object
- 分配 workspace
- 写 `<<<grid, block>>>`
- 定义 TensorRT/PyTorch binding

这些事情分别属于 device、ops、binding 或 test 层。

## tiny-cutlass 的 policy 命名

本目录的 primary policy type 应保持模板工厂风格：

```cpp
template <
    typename ArchTag,
    typename Element,
    typename ThreadblockShape0,
    typename ThreadblockShape1,
    typename WarpShape0,
    typename WarpShape1>
struct DefaultConv1x1ReluConv1x1;
```

不要把 dtype、layout、arch 写进 primary name。当前默认可能只实例化 `half + Sm80`
或者 `e4m3 + Sm89`，但选择应该留在 explicit instantiation、launcher、test 或 CMake
层，而不是藏进类名里。

