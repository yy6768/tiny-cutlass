# CUTLASS学习记（六）下：如果没有FlashAttention， Attention长什么样？
## 前言

- 论文终于结束了。超长停更应该也结束了。

- 本来我准备直接开始写online-softmax但直接按照原文的online-softmax所言实现感觉没有明显的实感（单个softmax语义太模糊了），所以为了完整的有“实现” flash attention的感觉，有了这篇naive attention的补充。
- 另外一个点就是写CUDA还是得profiling-guide。有一个baseline会比较好写。



## Attention Recap

标准的 Multi-Head Attention 分三步，每一步对应一个独立的 kernel：

$$O = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d}}\right) V$$

1. **Score**: $S = QK^T \cdot \text{scale}$，其中 $\text{scale} = 1/\sqrt{d}$；
2. **Softmax**: 对 $S$ 的每一行做 softmax(row-wise)； $P = Softmax(S)$
3. **Output**: $O = P \cdot V$。

 `examples/41_fused_multi_head_attention/kernel_forward.h`也是实现了上述这三步骤，只是完全融合成一个kernel了。

1. **Score / MM0**:  这里example 使用了标准的cutlass `mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum)` 计算当前 block 的 $QK^T$。 底层是最标准的sm80 GEMM（当前我是一个30/40系的RTX显卡，自动选择的）
2. **Online Softmax**: `iterative_softmax(...)` 是一个标准的online softmax，定义在 `kernel_forward.h:1153`。这个函数只维护每一行的 `mi`、`m_prime`、`s_prime` 和 `out_rescale`。 详细的内容我们会在后续的博客详细的说明。
3. **Output / MM1**: softmax 之后的当前 attention block 会通过 `MM0::B2bGemm::accumToSmem(...)` 写到 shared memory，然后 MM1 读取它并和 $V$ 做矩阵乘，`mma_pv(...)`  对应 $P \cdot V$。最后真正除以 softmax 分母是在 epilogue 里完成的：`EpilogueOutputOp rescale(s_prime, out_rescale)` 。



## Naive Attention：GPT 5.5的实现

按照前言说的，为了后续有一个明显的优化的baseline，我让GPT 5.5 帮我参照41 example的实现， 写一个完全分开的 “naive attention”的实现， 按照上述描述拆成3个独立的kernel的`naive_attention.cu`

和 example 41 的 fused 版本不同，这里中间矩阵 `P` 会真的写回 `block_P`，再被 softmax kernel 读回来。也就是说，这一版的 attention 流水线就是：

$$
S = QK^T  \rightarrow P=\text{softmax}(S) \rightarrow O = P\cdot V
$$



虽然经历了多次迭代（消耗了我不少token），我实现了不少

- MM0和MM1两个kernel 可以复用， 都是标准的GEMM
- Softmax 的实现可以使用CUTLASS相关的reduce函数

### MM0/MM1

对于我们的实现，我首先先把大部分代码直接复制过来。核心的MM0 定义可以当作当前 GEMM的参考实现。

example 41第一个关键的部分是 ` advance_to_block()` , 这个函数快速的定位到CTA计算职责对应的位置。

```
 CUTLASS_DEVICE bool advance_to_block() {
      auto bh      = blockIdx.z;
      auto head_id  = bh % num_heads;
      auto batch_id = bh / num_heads;
      auto row_id   = blockIdx.x;
      if (row_id >= num_queries) return false;
      p_ptr += batch_id * p_strideB + head_id * p_strideH + row_id * p_strideM;
      return true;
    }
```

此时仿照example41，使用是标准的DefaultGEMM Kernel, Epilogue换成DefaultGemm的epilogue存入Global Memory。

对应的CTA Level 的GEMM

```
  using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<...>;
  using Mma = typename DefaultGemm::Mma;
  using Epilogue = typename DefaultGemm::Epilogue;
```

- 对于MM0而言，主要的区别在于Epilogue。 41 example 使用 `MM0::B2bGemm::accumToSmem(...)` 将结果累加到SharedStorage里。后续会有一节专门研究CUTLASS的B2b算子。
- 对于MM1而言，主要区别在A矩阵（也就是P矩阵）的读入和Epilogue上（还要处理softmax的分母）。这个我们在对应章节在仔细研究。



### Softmax

这里我们按照最原始的实现， 选用3 pass 的 safe softmax，保证数值精度。下图来自于华盛顿cse599m

1.  第一部分计算整体的最大值
2. 求和
3. 除以分母归一化

![safe-softmax](https://typora-yy.oss-cn-hangzhou.aliyuncs.com/Typora-img/image-20260526003815990.png)



## 性能：NCU



