# CUTLASS学习记（七）上：Online Softmax

## 前言
- 接着上期naive attention继续学， 代码请参考
- 本篇主要参考了
  - [FlashAttention的原文](https://arxiv.org/pdf/2205.14135)
  - [从Online-Softmax到FlashAttention V1/V2/V3](https://zhuanlan.zhihu.com/p/668888063)




## 从 naive attention的 softmax 开始

上一篇中我们实现的 softmax 是最标准的 safe softmax。给定一行 score：

$$
s_1, s_2, \ldots, s_N
$$

为了数值稳定，先找最大值：

$$
m = \max_j s_j
$$

然后计算分母：

$$
d = \sum_j e^{s_j - m}
$$

最后归一化：

$$
p_j = \frac{e^{s_j - m}}{d}
$$

这个写法非常自然，但在 kernel 里就是 3 趟扫描：

1. 读一遍 `P`，找 `row_max`
2. 再读一遍 `P`，写回 `exp(x - row_max)`，同时求 `row_sum`
3. 再读一遍 `P`，写回 `P / row_sum`

对应的结构大概是：

```cpp
// Pass 1: row max
for (int j = tid; j < num_keys; j += kThreads) {
  mi = fmaxf(mi, row_ptr[j]);
}

// Pass 2: exp + row sum
for (int j = tid; j < num_keys; j += kThreads) {
  accum_t e = __expf(row_ptr[j] - row_max);
  row_ptr[j] = scalar_t(e);
  di += e;
}

// Pass 3: normalize
for (int j = tid; j < num_keys; j += kThreads) {
  row_ptr[j] = scalar_t(row_ptr[j] * inv_sum);
}
```

但如果从 CUDA kernel 的角度看，就有点难受：

- `P` 被反复读写
- max reduce 和 sum reduce 分两轮做
- 每一轮 cross-warp reduction 都要同步
- 对长序列来说，softmax 基本是在给 `P` 交 memory traffic 的税

`00b-profiling` 里 softmax 的 DRAM 压力和 `long_scoreboard` 就是这个结构的结果。

## Online Softmax 的核心

Online softmax 的关键点是：**max 和 denominator 可以一起维护**。

我们维护一个状态：

$$
(m, d)
$$

其中：

- `m` 是目前看过元素里的最大值
- `d` 是基于当前 `m` 的 softmax denominator

初始状态是：

$$
m = -\infty,\quad d = 0
$$

每看到一个新元素 `x`，更新规则是：

$$
m_{new} = \max(m, x)
$$

$$
d_{new} = d \cdot e^{m - m_{new}} + e^{x - m_{new}}
$$

这个公式的直觉其实很好理解：如果新的 max 变大了，那旧分母里的所有项都要按新的 max 重新缩放。

比如旧状态里我们保存的是：

$$
d = \sum e^{s_j - m}
$$

但现在 max 变成了 `m_new`，所以旧的分母要乘：

$$
e^{m - m_{new}}
$$

新元素自己的贡献就是：

$$
e^{x - m_{new}}
$$

这就是 online softmax。

## 为什么它适合并行归约

如果只是单线程扫一行，online recurrence 已经够用了。但我们的 softmax kernel 是 256 threads 一起处理一行 `P`，所以还需要合并两个部分状态。

假设两个 partial state 是：

$$
(m_0, d_0),\quad (m_1, d_1)
$$

合并后的 max 是：

$$
m = \max(m_0, m_1)
$$

合并后的 denominator 是：

$$
d = d_0 \cdot e^{m_0 - m} + d_1 \cdot e^{m_1 - m}
$$

这就是 `online_merge`：

```cpp
static CUTLASS_DEVICE void online_merge(
    accum_t& mi,
    accum_t& di,
    accum_t mj,
    accum_t dj) {

  if (di == accum_t(0)) {
    mi = mj;
    di = dj;
    return;
  }
  if (dj == accum_t(0)) {
    return;
  }

  accum_t m_new = fmaxf(mi, mj);
  di = di * __expf(mi - m_new) + dj * __expf(mj - m_new);
  mi = m_new;
}
```

这里两个 `if` 不是优化，而是为了处理初始空状态。初始时 `d = 0`，如果硬套公式，很容易把 `-inf` 和 `exp` 搅在一起，最后得到不想要的 NaN。

## 01 的代码结构

这次的实现文件是：

```text
csrc/flash-attention/01-online-softmax/
  online_softmax_attention.cu
  kernel_forward.h
  CMakeLists.txt
```

`online_softmax_attention.cu` 里仍然是三段式 testbed：

```cpp
using MM0 = AttentionMMKernel<...>;
using Softmax = AttentionSoftmaxKernel<scalar_t, 256>;
using MM1 = AttentionMMKernel<...>;
```

设备端 buffer 也没有变化：

```cpp
block_Q  // [B, Sq, H, d]
block_K  // [B, Sk, H, d]
block_V  // [B, Sk, H, dv]
block_P  // [B, H, Sq, Sk]
block_O  // [B, Sq, H, dv]
```

这很重要：`01` 仍然有 `block_P`。

所以它的 launch 顺序还是：

```text
MM0 -> Softmax -> MM1
```

这一篇只改 `AttentionSoftmaxKernel`。

## online_warp_reduce

`00` 里有两个不同的 warp reduction：

- `warp_reduce_max`
- `warp_reduce_sum`

`01` 里变成一个：

```cpp
static CUTLASS_DEVICE void online_warp_reduce(accum_t& mi, accum_t& di) {
  CUTLASS_PRAGMA_UNROLL
  for (int o = kWarpSize / 2; o > 0; o >>= 1) {
    accum_t mj = __shfl_xor_sync(0xffffffff, mi, o);
    accum_t dj = __shfl_xor_sync(0xffffffff, di, o);
    online_merge(mi, di, mj, dj);
  }
}
```

每次 shuffle 需要传两个 float：`m` 和 `d`。

这个代价是有的，但它换掉的是一整趟 global memory 扫描。对于 attention 这种一行 `P` 很长的场景，这个交换是值得的。

## attention_kernel：从 3-pass 到 2-pass

`AttentionSoftmaxKernel::attention_kernel` 现在只做两趟。

第一趟：每个线程 stride 读取自己负责的元素，并在线维护 `(m, d)`：

```cpp
accum_t mi = -cutlass::platform::numeric_limits<accum_t>::infinity();
accum_t di = accum_t(0);

for (int j = tid; j < p.num_keys; j += kThreads) {
  accum_t x = accum_t(row_ptr[j]);
  online_merge(mi, di, x, accum_t(1));
}
```

然后做 warp 内归约：

```cpp
online_warp_reduce(mi, di);
if (lane_id == 0) {
  warp_storage_m[warp_id] = mi;
  warp_storage_d[warp_id] = di;
}
__syncthreads();
```

再让 warp 0 做 cross-warp reduction：

```cpp
if (warp_id == 0) {
  mi = (lane_id < kNumWarps)
      ? warp_storage_m[lane_id]
      : -cutlass::platform::numeric_limits<accum_t>::infinity();
  di = (lane_id < kNumWarps) ? warp_storage_d[lane_id] : accum_t(0);

  online_warp_reduce(mi, di);
  if (lane_id == 0) {
    warp_storage_m[0] = mi;
    warp_storage_d[0] = di;
  }
}
__syncthreads();
```

第二趟：用最终的 `row_max` 和 `inv_sum` 写回归一化结果：

```cpp
accum_t row_max = warp_storage_m[0];
accum_t inv_sum = accum_t(1) / warp_storage_d[0];

for (int j = tid; j < p.num_keys; j += kThreads) {
  row_ptr[j] = scalar_t(__expf(accum_t(row_ptr[j]) - row_max) * inv_sum);
}
```

结构上很清楚：

```text
Pass 1: read P, online reduce (m, d)
Pass 2: read P, write normalized P
```

相比 `00`：

```text
00: read P -> read/write P -> read/write P
01: read P -> read/write P
```

## Shared Memory 的变化

`00` 的 softmax 每次 reduction 只需要存一个 float 数组：

```text
kNumWarps * sizeof(float)
```

`01` 需要同时存 `m` 和 `d`：

```cpp
CUTLASS_HOST int getSmemBytes() const {
  return 2 * kNumWarps * sizeof(accum_t);
}
```

对于 `kThreads = 256`：

```text
kNumWarps = 8
smem = 2 * 8 * 4 = 64 bytes
```

这个 shared memory 增量可以忽略。真正的差异还是少了一趟对 `P` 的 global memory 访问。

## 这版到底改了什么

没改的部分：

- `MM0` 仍然是 CUTLASS threadblock GEMM
- `MM1` 仍然是 CUTLASS threadblock GEMM
- `P` 仍然是 `[B, H, Sq, Sk]`
- `Testbed`、初始化、host reference、timing 流程基本沿用 `00`
- grid 仍然是一行 `P` 对应一个 softmax block

改了的部分：

- safe softmax 从 3-pass 改成 2-pass
- `warp_reduce_max` / `warp_reduce_sum` 合并成 online reduction
- shared memory 存 `m` 和 `d`
- softmax kernel 少一轮读写和同步

这就是 `01` 的全部边界。

## 性能应该怎么看

这里先不要把 `01` 吹成 FlashAttention。

它能改善的是 softmax kernel 内部的访存结构：

| 项目 | 00 safe softmax | 01 online softmax |
|---|---|---|
| 扫描次数 | 3 pass | 2 pass |
| global read | 3 次 | 2 次 |
| global write | 2 次 | 1 次 |
| reduction | max 和 sum 分开 | `(m, d)` 一起 |
| `P` 是否还在 HBM | 是 | 是 |

所以预期是：

- softmax kernel 的 DRAM 压力下降
- softmax kernel 的 barrier / scoreboard stall 有机会下降
- 总体 attention 时间会改善，但不会发生质变

真正的质变要等 `02`：MM0 的结果不再写完整 `P`，而是在 tile 里直接进入 online softmax 和 MM1。

## 为什么这篇仍然值得写

`01` 很像一个过渡版本。

从最终目标看，它不够激动人心，因为完整 `P` 还在。但从学习路线看，它非常必要，因为它把后面 fused attention 里的核心状态先拆出来了：

```text
m        当前已经看过的最大值
d / l    当前 softmax denominator
merge    两段 partial softmax 状态的合并
```

这些东西到了 `02` 会变成 tile 级别的：

```text
for each K/V tile:
  S_tile = Q_tile @ K_tile^T
  update m/l
  rescale old O
  add P_tile @ V_tile
```

如果没有 `01` 这一步，`02` 的 `iterative_softmax(...)` 就会显得太突然。

## 当前结论

`01-online-softmax` 的定位应该很明确：

- 它不是 FlashAttention
- 它没有消除完整 `P`
- 它只把 softmax 从 3-pass 改成 2-pass
- 它的核心价值是验证 online softmax 的数值递推和并行归约

下一篇真正重要的事情，是把 `P` 从 global memory 里拿掉。

这才是 `02-tiled-online-attention` 要解决的问题。
