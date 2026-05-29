# CUTLASS学习记（七）：把P矩阵拿掉以后

## 前言

上一篇 naive attention 的重点是建立 baseline：

$$
S = QK^T \rightarrow P=\text{softmax}(S) \rightarrow O=PV
$$

这个写法足够直观，但是它会把完整的 `P` 写到 global memory。`00-naive-attention` 里 `block_P` 的布局是 `[B, H, Sq, Sk]`，所以当序列长度变大时，`P` 很快就会成为 attention 里最显眼的中间状态。

`01-online-softmax` 只是把 softmax kernel 从 3-pass safe softmax 换成 online softmax。它仍然需要：

1. MM0 写完整 `P`
2. softmax 读写 `P`
3. MM1 再读 `P`

所以 01 的结论比较清楚：online softmax 本身不是重点。真正的重点是让 `P` 不再作为完整矩阵存在。

## 02 做了什么

`examples/flash_attention/02-tiled-online-attention` 现在使用 CUTLASS example 41 的 fused forward kernel。它不是手写 scalar CUDA kernel，而是直接走 CUTLASS/tensor core 路径：

```text
MM0 CUTLASS MMA
  -> iterative online softmax
  -> B2B shared-memory handoff
  -> MM1 CUTLASS MMA
  -> normalize/write O
```

本地文件结构是：

```text
02-tiled-online-attention/
  kernel_forward.h          // 本地化的 example 41 fused forward kernel
  tiled_online_attention.cu // 保留 00/01 风格的测试入口
  CMakeLists.txt
```

`tiled_online_attention.cu` 里仍然使用和 00/01 类似的命令行参数、BMHK tensor 初始化和 host reference。区别是设备端不再分配 `block_P`。主路径只有：

```cpp
block_Q
block_K
block_V
block_O
```

这里的 02 对 example 41 做了一个局部调整：跨 key tile 的中间输出用 `output_t` 保存，epilogue 内部缩放仍然用 `float` 计算。这样大 `d_v` 路径可以避免额外的 float `output_accum_ptr` buffer。这个 buffer 不是 attention matrix `P`，但它确实会带来额外 global memory 流量。完整的 `[B, H, Sq, Sk]` 矩阵仍然没有回到 global memory。

## 核心代码路径

02 的 kernel 类型是：

```cpp
using Attention = AttentionKernel<
    cutlass::half_t,
    cutlass::arch::Sm80,
    true,
    kQueriesPerBlock,
    kKeysPerBlock,
    kMaxK,
    false,
    false>;
```

launch 的是：

```cpp
attention_kernel_batched_impl<Attention>
```

这对应 example 41 的 `kernel_forward.h`。关键路径可以按 attention 语义分成三段。

### MM0: score tile

MM0 使用 CUTLASS threadblock MMA 计算当前 query block 和 key block 的 score：

```cpp
mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);
```

这里的 `accum` 不是写回 global memory 的 `P`，而是当前 CTA 内部的 accumulator fragment。它代表当前 tile 的：

$$
Q_{tile}K_{tile}^T
$$

### Online Softmax

MM0 之后直接调用：

```cpp
iterative_softmax(...)
```

这个函数在 accumulator 上更新每一行的 softmax 状态：

```text
mi       当前已经看过的最大 score
m_prime  上一次用于缩放 output accumulator 的 max
s_prime  softmax denominator
out_rescale  output accumulator 的缩放因子
```

数学上对应：

$$
m_{new} = \max(m_{old}, m_{tile})
$$

$$
l_{new} = l_{old}\exp(m_{old}-m_{new}) + \sum_j \exp(s_j-m_{new})
$$

同时当前 tile 的 score 会被改成未归一化的 attention 权重：

$$
\exp(s_j - m_{new})
$$

### MM1: attention tile 乘 V

softmax 后的当前 attention tile 会通过：

```cpp
MM0::B2bGemm::accumToSmem(...)
```

写到 shared memory。然后 MM1 从 shared memory 读取这个 tile，和当前 value tile 做 CUTLASS MMA：

```cpp
mma_pv(gemm_k_iterations, accum_o, iterator_V, accum_o);
```

最后 epilogue 使用 `s_prime` 和 `out_rescale` 做归一化，把结果写到 `O`。

这就是 02 和 00/01 的本质差异：`P` 只在 tile 级别以 register/shared memory 的形式短暂存在，不再作为完整矩阵写回 global memory。

## 性能观察

测试均使用 Release exe。

### d = 64

参数：

```text
Sq=1024, Sk=1024, d=64, d_v=64, H=32, B=1
```

结果：

```text
00 naive attention   约 1.24 ms
02 fused attention   约 0.33 ms
```

这里 02 明显更快，因为 MM0/MM1 都走 CUTLASS MMA，同时避免了完整 `P` 的 global memory 生命周期。

### d = 128

参数：

```text
Sq=1024, Sk=1024, d=128, d_v=128, H=32, B=1
```

结果：

```text
00 naive attention   约 1.38 ms
02 fused attention   约 0.71 ms
```

02 仍然明显更快。

### d = 256

参数：

```text
Sq=1024, Sk=1024, d=256, d_v=256, H=32, B=1
```

结果：

```text
01 online softmax    约 1.76 ms
02 fused attention   约 1.57 ms
```

这个参数下最终使用的 tile 是：

```text
kQueriesPerBlock = 64
kKeysPerBlock    = 64
kMaxK            = 65536
```

之前直接沿用 example 41 的 `d_v > 128` 路径时，`output_accum_t = float`，会分配 `output_accum_ptr` 并在多个 key tile 之间读写 float 中间结果。当前 02 把中间输出类型改成 `output_t`，同时保留 epilogue 的 float 缩放计算，减少了这部分 global memory 压力。`64x64` 的 tile 在目标参数下也比 `32x64` 更好。

## 当前结论

02 现在已经不是“语义学习版 scalar kernel”，而是 CUTLASS/tensor core fused attention 路径：

```text
AttentionKernel
  MM0::Mma
  iterative_softmax
  MM0::B2bGemm::accumToSmem
  MM1::Mma
  MemoryEfficientAttentionNormalize
```

它在 `d=64/128/256` 下都已经通过 correctness check，并且在目标参数 `Sq=Sk=1024, d=d_v=256, H=32, B=1` 上快于 01。下一步可以继续用 Nsight Compute 看 `64x64` tile 下的 occupancy、shared memory、epilogue 和 global memory 行为。
