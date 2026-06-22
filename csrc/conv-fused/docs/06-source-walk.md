# 06-source-walk: 按源码走一遍 SM80 RF fused conv

## 前言

前面几篇把分层讲完了，但还需要一篇非常具体的走读：从
`fused_two_convs_f16_sm80_rf.cu` 入口开始，看到 `device`、`kernel`，最后落到
RF-resident threadblock mainloop。这样后面改 `tiny-cutlass/csrc/conv-fused`
时，我们知道每一层应该像谁。

## 入口只选择类型

`fused_two_convs_f16_sm80_rf.cu` 的 fused 路径从
`run_fused_conv2d_fprop_optimized_f16_sm80_rf_res()` 开始。这里没有写 kernel body，
只做四件事：

1. 选择 element type：`half_t`。
2. 选择 tile：`ThreadblockShape0/1`、`WarpShape0/1`、`InstructionShape`。
3. 选择两个 epilogue output op。
4. 调 `DefaultB2bConv2dFprop<...>::Kernel` 生成 kernel type。

关键点是最后一步：

```cpp
using B2bConv2dFpropKernel =
    typename cutlass::conv::kernel::DefaultB2bConv2dFprop<...>::Kernel;
using B2bConv2dFprop =
    cutlass::conv::device::B2bImplicitGemmConvolution<B2bConv2dFpropKernel>;
```

所以官方路径不是“写一个 `run_xxx` 函数直接 launch”，而是先得到一个 kernel type，
再交给 CUTLASS-style device operator。

## device operator 是 host contract

`device/b2b_implicit_gemm_convolution.h` 是 host 侧 contract。它有几个必须保留的
形状：

- `Arguments`: problem size、TensorRef、epilogue params。
- `can_implement`: 先问 iterator 能不能实现，再检查 B2B fusion 的 shape 约束。
- `initialize`: 把 `Arguments` 编译成 kernel `Params`，并设置 dynamic shared memory。
- `run`: 计算 grid/block/smem，调用 `cutlass::Kernel<Kernel><<<...>>>`。
- `operator()`: 串起 `initialize` 和 `run`。

`can_implement` 里最能说明 B2B conv 约束的是这些检查：

```text
problem_size_0.m() == problem_size_1.m()
problem_size_0.n() == problem_size_1.k()
problem_size_1.R == 1 && problem_size_1.S == 1
problem_size_0.n() <= ThreadblockShape0::kN
problem_size_1.n() <= ThreadblockShape1::kN
```

这也是为什么普通 `2x2 stride2 pool` 不能直接塞进这个 family：pool 会改变空间 M，
而 example 13 的 B2B conv 要求两段 implicit GEMM 的 M 一致。

## kernel policy 决定 RF 还是 SMEM

`kernel/default_b2b_conv2d_fprop_sm80.h` 的 RF 路径里，最重要的类型不是
`DefaultMmaCore` 本身，而是 A1 的来源：

```cpp
using FragmentIteratorA1 =
    cutlass::gemm::warp::MmaTensorOpFragmentIterator<...>;
```

然后它把这个 iterator 交给：

```cpp
using B2bMma = threadblock::B2bImplicitGemmMultistage<...>;
```

这就决定了 A1 不是从 global memory 读，也不是从 shared accumulator 读，而是从
第一段 `accum0` fragment 读。相对地，SMEM accumulator 版本会额外定义
`SmemIteratorD0` 和 `WarpIteratorA1`，再使用
`B2bImplicitGemmMultistageSmemAccumulator`。

所以以后我们看一个实现是不是 RF fused，不看名字里有没有 `fused`，要看 A1 operand
的 iterator 到底从哪里来。

## threadblock mainloop 才是真融合

`threadblock/b2b_implicit_gemm_multistage.h` 里，第一段 mainloop 得到 `accum0` 后，
第二段开始时构造：

```cpp
FragmentIteratorA1 warp_tile_iterator_A1_(accum0);
```

随后加载 A1 fragment 时顺手应用第一段 epilogue：

```cpp
warp_tile_iterator_A1_.load(
    warp_loaded_frag_A1[0],
    warp_loaded_frag_A1_scale[0],
    warp_loaded_frag_A1_bias[0],
    output_op_0);
```

主循环里也一样：

```cpp
warp_tile_iterator_A1_.load(
    warp_loaded_frag_A1[(warp_mma_k + 1) % 2],
    warp_loaded_frag_A1_scale[(warp_mma_k + 1) % 2],
    warp_loaded_frag_A1_bias[(warp_mma_k + 1) % 2],
    output_op_0);
```

这里 `output_op_0` 可以是 relu、scale、bias、quantize 这类逐元素转换。它发生在
accumulator fragment 变成第二段 A operand 的过程中，而不是一个单独的 global
store/load 阶段。

## tiny-cutlass 当前应该怎样贴近它

`csrc/conv-fused` 现在应该保持这个对应关系：

- `kernel/conv1x1_relu_conv1x1.h`: 只组装 `DefaultB2bConv2dFprop<...>::Kernel`。
- `device/conv1x1_relu_conv1x1.{h,cu}`: family-specific device object，直接包
  `cutlass::conv::device::B2bImplicitGemmConvolution<Kernel>`。
- `ops/conv1x1_relu_conv1x1.{h,cu}`: raw pointer API 和 problem descriptor。
- `threads/epilogue_ops.h`: 只放 thread-level epilogue aliases。

不应该再有本地复刻的 generic device wrapper，也不应该有 staged global-memory
pool path 假装成 fusion。后续如果做 pool，需要先在 threadblock/RF/SMEM 这一层写清楚
pool 的 operand residency，而不是从 binding 或 test 入口开始堆代码。
