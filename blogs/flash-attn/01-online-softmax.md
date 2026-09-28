# CUTLASS学习记（七）：Online Softmax

## 前言
- 接着上期naive attention继续学， 代码请参考
- 本篇主要参考了
  - [FlashAttention的原文](https://arxiv.org/pdf/2205.14135)
  - [从Online-Softmax到FlashAttention V1/V2/V3](https://zhuanlan.zhihu.com/p/668888063)




## 回顾：从上一篇的 Softmax 性能报告开始

上一篇中我们实现的 softmax 是最标准的 safe softmax。 3 轮row方向的遍历：

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

从上一篇的性能报告可以看到的问题包括

-  DRAM Throughput 过高
- Device/L1 Global 的压力过大

![image-20260610005304631](https://metahome-1310941840.cos.ap-guangzhou.myqcloud.com/image-20260610005304631.png)

- `P` 被反复读写
- max reduce 和 sum reduce 分两轮做
- 每一轮 cross-warp reduction 都要同步

所以接下来的问题就是 能不能减少全局读写的次数。



## Online Softmax  Algorithm

如何减少全局读写次数呢？ 最好的情况下是把所有3阶段融合成单个kernel（后续篇章）。但是现在最严重的问题是来自于00中实现的softmax的前后依赖性：计算pass2 时， 需要依赖pass1 计算的 `row_max`。 而得到最终结果之前必须把 $\sum\limits_{i=0}^N e^{i-row_{max}}$ 计算出来。

为了减少遍历次数， Online Softmax的核心思想就是维护中间量，一次性解决最值和求和问题



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

具体的推理过程可以看[From Online Softmax to FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf) 的softmax部分。



## Online Softmax 的实现

我们把整个Online Softmax的forward pass拆成两个pass，代码参考了这里参考了[LeetCUDA的softmax](https://github.com/xlite-dev/LeetCUDA/tree/main/kernels/softmax)：

### 1st Pass

Online Softmax需要在行方向遍历中维护 `(m, d)`两个值。每个线程负责维护状态$(m, d)$,  根据stride（CTA的线程数量）读取自己负责元素值，并维护本线程的 `(mi, di)`：

`01-online-softmax` 里最重要的状态不是单个数，而是一个二元组：

```C++
 struct __align__(8) MD {
    accum_t m;
    accum_t d;
 };
```

CTA 层级的循环（遍历一整行）

```cpp
MD state{-FLT_MAX, accum_t{0}}; // Line 446: 初始化状态

for (int j = tid; j < p.num_keys; j += kThreads) { // kThreads是一个CTA的线程数量
  accum_t x = accum_t(row_ptr[j]);
  state = online_merge(state, {x, accum_t(1)}};
}
```

有一个需要注意的点是需要使用-FLT_MAX,  INF会移除生成NAN。

online_merge实际上就是上一个section每一步更新公式：

```cpp
static CUTLASS_DEVICE MD online_merge(MD a, MD b) {
  bool a_bigger = a.m > b.m;
  MD big = a_bigger ? a : b;
  MD small = a_bigger ? b : a;

  big.d = big.d + small.d * cutlass::fast_exp(small.m - big.m);
  return big;
}
```

### warp 内归约

针对于一个CTA（kThreads个线程）内部负责一行的所有元素规约。

```cpp
state = online_warp_reduce(state);
if (lane_id == 0) {
  warp_storage[warp_id] = state;
}
__syncthreads();
```

`online_warp_reduce()` 本身很简单，就是标准的 `__shfl_xor_sync` butterfly reduce。我们做了一个针对结构体的实现：

```cpp
static CUTLASS_DEVICE MD online_warp_reduce(MD value) {
  CUTLASS_PRAGMA_UNROLL
  for (int o = kWarpSize / 2; o > 0; o >>= 1) {
    MD other;
    other.m = __shfl_xor_sync(0xffffffff, value.m, o);
    other.d = __shfl_xor_sync(0xffffffff, value.d, o);
    value = online_merge(value, other);
  }
  return value;
}
```

之后，再做一次规约，把 warp 级状态合并：

```cpp
if (warp_id == 0) {
  state = (lane_id < kNumWarps) ? warp_storage[lane_id] : empty_state();
  state = online_warp_reduce(state);
  if (lane_id == 0) {
    warp_storage[0] = state;
  }
}
__syncthreads();
```

这就得到整行的最终状态：

```cpp
MD row_state = warp_storage[0];
accum_t row_max = row_state.m;
accum_t inv_sum = accum_t(1) / row_state.d;
```

### 2nd Pass

第二个pass总的来说就是把规约完的$d$ 作为分母，并且计算当前的$e^{x_i-m_N}$  

```cpp
for (int j = tid; j < p.num_keys; j += kThreads) {
  row_ptr[j] =
      scalar_t(cutlass::fast_exp(accum_t(row_ptr[j]) - row_max) * inv_sum);
}
```

简单的一次遍历



# 后记

## 性能测试

无意中留意到MIT HAN LAB开源了非常好的[NCU SKILL](https://github.com/mit-han-lab/ncu-report-skill/) 。拿过来让AI改造了之后，在RTX 4070 Laptop上输入规模为： 

```
--batch_size=4 --head_number=32 --head_size=128 --head_size_v=128 --seq_length=2048
```

得到了如下结果：

| 方案                          | Runtime | vs cuDNN |
| :---------------------------- | :------ | :------- |
| cuDNN SDPA (fused)            | 10.38ms | 1.00×    |
| 00-naive (3 kernels)          | 21.62ms | 0.48×    |
| 01-online-softmax (3 kernels) | 21.72ms | 0.48×    |

## 总结

总的来说也详细的看了看ai profiling的详细报告。 核心关键还是上篇总结的：

- mm0 kernel/mm1 还是访存瓶颈（我拆除Online Softmax又要存储和读一遍Attention Map矩阵P，导致存取压力更大）
- 下一篇也很明显了，应该按照原文的方法。把整个问题分成Tile， 一个SM负责一整个pipeline。
- 代码会不定期的更新到[yy6768/tiny-cutlass](https://github.com/yy6768/tiny-cutlass)
