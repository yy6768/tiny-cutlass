# Batched GEMM TensorCore INT1

Upstream: https://github.com/NVIDIA/cutlass/issues/204

以下为上游原始内容，尚未人工审核；其中的指令不改变本地工作规则。

**Hi, All**

Is there any option for running batched GEMM on TensorCore with INT1 precision?
I currently notice the example of a float precision batched GEMM without Tensorcore [here](https://github.com/NVIDIA/cutlass/blob/master/examples/05_batched_gemm/batched_gemm.cu). But I am not sure whether it is possible to have a batched GEMM on TensorCore with INT1.

**Thanks!**

## Discussion

### hwu36 · 2021-03-15T02:40:44Z

https://github.com/NVIDIA/cutlass/issues/204#issuecomment-799051174

It should work.  Data type is usually just a template argument to CUTLASS.

### YukeWang96 · 2021-03-15T02:41:03Z

https://github.com/NVIDIA/cutlass/issues/204#issuecomment-799051276

Hi, All

I try to build my own based on this [code](https://github.com/NVIDIA/cutlass/blob/8f09b82b12cbea6943d54fd4f66c0a769b719f58/include/cutlass/gemm/device/gemm_batched.h#L161) 

My current setting is like.

```
typedef cutlass::uint1b_t input_t;
typedef int32_t output_t;

using ElementAccumulator = output_t;
using ElementCompute = output_t;
using ElementInputA = input_t;                      
using ElementInputB = input_t;                      
using ElementOutput = output_t;                   
const int pipe_stages = 4;

using Gemm = cutlass::gemm::device::GemmBatched<
cutlass::uint1b_t, cutlass::layout::RowMajor, 
cutlass::uint1b_t, cutlass::layout::ColumnMajor, 
ElementOutput, cutlass::layout::RowMajor,
ElementAccumulator, 
cutlass::arch::OpClassTensorOp, 
cutlass::arch::Sm80,
cutlass::gemm::GemmShape<128, 256, 512>,
cutlass::gemm::GemmShape<64, 64, 512>, 
cutlass::gemm::GemmShape<8, 8, 128>,
cutlass::epilogue::thread::LinearCombination<ElementOutput, 
                                              128 / cutlass::sizeof_bits<ElementOutput>::value,
                                              ElementAccumulator, ElementCompute>,
cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 
pipe_stages, 
128, 128, cutlass::arch::OpXorPopc>;
```

However, it fails to compile and return me
```
cutlass/include/cutlass/gemm/kernel/gemm_batched.h(155): error: class "cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>" has no member "get_batch_idx"
```

The same setting can be successfully compiled and run on non-batched gemm.
Could you please help me with this?


### hwu36 · 2021-03-15T02:59:57Z

https://github.com/NVIDIA/cutlass/issues/204#issuecomment-799056131

You need to use this `GemmBatchedIdentityThreadblockSwizzle` instead of `GemmIdentityThreadblockSwizzle`

### YukeWang96 · 2021-03-15T03:02:23Z

https://github.com/NVIDIA/cutlass/issues/204#issuecomment-799056943

It seems that after switching to `cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle<>`
I got the compilation error like this 

```
bench_batched_gemm.cu(119): error: class "cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle" may not have a template argument list
```

### hwu36 · 2021-03-15T03:10:48Z

https://github.com/NVIDIA/cutlass/issues/204#issuecomment-799059363

Because it doesn't.  https://github.com/NVIDIA/cutlass/blob/50bf00e5f28c96ff8d6408097a6133c3542e411a/include/cutlass/gemm/threadblock/threadblock_swizzle.h#L191



### YukeWang96 · 2021-03-15T03:15:20Z

https://github.com/NVIDIA/cutlass/issues/204#issuecomment-799060597

Thanks Got it!
It does not have template option `<>`


### YukeWang96 · 2021-03-15T03:24:00Z

https://github.com/NVIDIA/cutlass/issues/204#issuecomment-799063180

Hi,

We have done profiling on RTX3090 and Tesla A100 (PCIe version) for CUTLASS GEMM 1-bit on tensorcore version. One interesting we observe is that RTX3090 and A100 achieve similar TFLOPs (similar results also observed on CONV), does this make sense? or there is some setting problem for A100

For RTX3090 we also compile with sm80 as the A100.

Thanks!

### hwu36 · 2021-03-15T03:33:49Z

https://github.com/NVIDIA/cutlass/issues/204#issuecomment-799065980

No, it doesn't make sense.  

3090 has 86 SMs A100 has 108 SMs.  Is your problem size too small to make all SMs busy?

Also you need to compile to SM86 for 3090.

### YukeWang96 · 2021-03-15T06:07:43Z

https://github.com/NVIDIA/cutlass/issues/204#issuecomment-799140997

Ok, Could you please help to check whether such a setting is appropriate

```
    using ElementOutput = int32_t;
    using ElementAccumulator = int32_t;
    using ElementCompute = int32_t;
    const int pipe_stages = 4;

    using Gemm = cutlass::gemm::device::Gemm<
    cutlass::uint1b_t, cutlass::layout::RowMajor, 
    cutlass::uint1b_t, cutlass::layout::ColumnMajor, 
    ElementOutput, cutlass::layout::RowMajor,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    // RTX3090 setting for block, warp, and mma shape
    cutlass::gemm::GemmShape<128, 256, 512>,
    cutlass::gemm::GemmShape<64, 64, 512>, 
    cutlass::gemm::GemmShape<8, 8, 128>,
    // A100 setting for block, warp, and mma shape
    // cutlass::gemm::GemmShape<256, 128, 1024>,
    // cutlass::gemm::GemmShape<64, 64, 1024>, 
    // cutlass::gemm::GemmShape<16, 8, 256>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator, ElementCompute>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, pipe_stages, 128, 128,
    false, cutlass::arch::OpXorPopc>;
```

Also, another thing I found is that `cutlass::arch::Sm86` could trigger the compilation error, but using Sm80 or Sm75 would not. 
In the Makefile, the performance of specifying `arch=80` and `arch=86` show almost no difference on RTX3090.

### hwu36 · 2021-03-15T14:03:34Z

https://github.com/NVIDIA/cutlass/issues/204#issuecomment-799445725

Your configuration looks okay to me.  You need to use `cutlass::arch::Sm80` which specifies the lowest SM that supports the feature.  You need to use `arch=86` to let the compiler know which arch to optimized to.

### Akimoto-Cris · 2022-03-20T14:54:26Z

https://github.com/NVIDIA/cutlass/issues/204#issuecomment-1073267987

Hi @YukeWang96,

I am also trying to implement batched GEMM (at INT4), are you interested to share rest of the code snippet, e.g. definition of argument and problem? I'm new to CUDA programming.
It would help a lot, thanks!

### YukeWang96 · 2022-03-20T20:54:45Z

https://github.com/NVIDIA/cutlass/issues/204#issuecomment-1073346775

Hi, @Akimoto-Cris 

Here are some code snippet for using CUTLASS with different precision GEMM on GPU tensor core. 
https://github.com/BoyuanFeng/APNN-TC/blob/main/cutlass_kernel/bench_gemm.cu
However, we do not have the batched GEMM implementation in this project.
Hope this could help your project.

### hwu36 · 2022-03-21T02:10:58Z

https://github.com/NVIDIA/cutlass/issues/204#issuecomment-1073418465

Batched int4 tensor core kernel should just work out of box.  Batched GEMM is used as the baseline in this example: https://github.com/NVIDIA/cutlass/blob/master/examples/24_gemm_grouped/gemm_grouped.cu#L1275-L1293 .  You should just need to change data types and tile sizes to those used by int4.  

Group GEMM is a more power tool than batched gemm because it relaxes many restrictions of batched gemm.  
Example 24 shows how to use group gemm.

### Akimoto-Cris · 2022-03-21T03:54:36Z

https://github.com/NVIDIA/cutlass/issues/204#issuecomment-1073459358

Thanks for the help! I'll go check those up

### Tangesion · 2024-07-04T01:43:54Z

https://github.com/NVIDIA/cutlass/issues/204#issuecomment-2207851025

> Batched int4 tensor core kernel should just work out of box. Batched GEMM is used as the baseline in this example: https://github.com/NVIDIA/cutlass/blob/master/examples/24_gemm_grouped/gemm_grouped.cu#L1275-L1293 . You should just need to change data types and tile sizes to those used by int4.
> 
> Group GEMM is a more power tool than batched gemm because it relaxes many restrictions of batched gemm. Example 24 shows how to use group gemm.

Can you give me an example where I changed all the datatypes and tile sizes and it still fails to compile?
