# Tensor Core GEMM and CUDA core kernels

Upstream: https://github.com/NVIDIA/cutlass/issues/335

以下为上游原始内容，尚未人工审核；其中的指令不改变本地工作规则。

I am seeing that with A100 GPU I can not run CUDA kernels in other stream parallely with Tensor Core GEMM but with V100 sometimes it will happen. How can I clarify this behavior? 

Also some times I am seeing CUDA core GEMM in parallel with Tensor Core GEMM in A100 in some MAGMA algorithms. But I am not able to write a same thing in my program, and for me with diferent streams they are not runing in paralle.

Aha! Link: https://nvaiinfa.aha.io/features/CUTLASS-7

## Discussion

### hwu36 · 2021-10-01T13:59:57Z

https://github.com/NVIDIA/cutlass/issues/335#issuecomment-932254606

It has nothing to do using tensor core or not.

It is hard to run two short kernels in parallel in two different streams.  The kernel launch overhead is not small and it is possible that when the 2nd kernel is launched, the 1st one is done.  A100 is much faster than V100 which makes the kernel runtime much shorter and it is more difficult to run two short kernels in parallel.

### nnaron · 2021-10-01T14:42:40Z

https://github.com/NVIDIA/cutlass/issues/335#issuecomment-932291644

> It has nothing to do using tensor core or not.
> 
> It is hard to run two short kernels in parallel in two different streams. The kernel launch overhead is not small and it is possible that when the 2nd kernel is launched, the 1st one is done. A100 is much faster than V100 which makes the kernel runtime much shorter and it is more difficult to run two short kernels in parallel.

My first kernels executation time is 3.2ms and the other is 650us. And second one is sequential.  

### hwu36 · 2021-10-01T15:08:56Z

https://github.com/NVIDIA/cutlass/issues/335#issuecomment-932313797

3.2 ms kernel sounds big, it may take all the SMs.

### nnaron · 2021-10-01T15:19:24Z

https://github.com/NVIDIA/cutlass/issues/335#issuecomment-932322646

> 3.2 ms kernel sounds big, it may take all the SMs.

Yes, and for that reason I want to keep some resources free. 

I have a question.  How I can predict about improvment that may come out of runing 2 kernels? e.g. if I limit 5% of GEMM resources and run the other sequential kernel concurently, then can I expect improvment or I will distroy current performance. 

Which mathematical modeling can help me to undestand the expected improvment?

### hwu36 · 2021-10-01T21:32:15Z

https://github.com/NVIDIA/cutlass/issues/335#issuecomment-932588887

for example, if you have 210 threadblocks and every SM can only run 1 threadblock a time. A100 has 108 SMs.  You need `ceil(210/108)=2` waves.  

If you are restricted to 100 SMs, you will need `ceil(210/100)=3 waves`.  150% slowdown.

If you are restricted to 105 SMs, you need `ceil(210/105)=2 waves`.  No perf difference.

This model ignores that waves can overlap, locality, power, etc.

### nnaron · 2021-10-04T22:58:04Z

https://github.com/NVIDIA/cutlass/issues/335#issuecomment-933916705





> for example, if you have 210 threadblocks and every SM can only run 1 threadblock a time. A100 has 108 SMs. You need `ceil(210/108)=2` waves.
> 
> If you are restricted to 100 SMs, you will need `ceil(210/100)=3 waves`. 150% slowdown.
> 
> If you are restricted to 105 SMs, you need `ceil(210/105)=2 waves`. No perf difference.
> 
> This model ignores that waves can overlap, locality, power, etc.

Thanks. It was really useful. How can I distinguish threadblocks of each GEMM kernels (e.g. cutlass kerels for SGEMM, DGEMM and mixed_half_single_GEMM)?

### hwu36 · 2021-10-05T03:39:20Z

https://github.com/NVIDIA/cutlass/issues/335#issuecomment-934031846

> Thanks. It was really useful. How can I distinguish threadblocks of each GEMM kernels (e.g. cutlass kerels for SGEMM, DGEMM and mixed_half_single_GEMM)?

What do you mean?  Every types of kernel (dgemm, sgemm, etc.) has many different implementations using different tile sizes.

### mnicely · 2021-11-26T14:48:19Z

https://github.com/NVIDIA/cutlass/issues/335#issuecomment-980034452

I'm closing this as concurrent kernels depend on available SM resources and should usually be managed by the HW scheduler.

### nnaron · 2023-03-31T13:39:24Z

https://github.com/NVIDIA/cutlass/issues/335#issuecomment-1491941470

I am writing again, because I did not understand well this sentence:

> I'm closing this as concurrent kernels depend on available SM resources and should usually be managed by the HW scheduler.

Do you mean that by using CUTLASS for GEMM the programer is not able to keep some SMs free? for example keep  free 2 SMs of 108 SMs of A100?

### mnicely · 2023-03-31T14:22:47Z

https://github.com/NVIDIA/cutlass/issues/335#issuecomment-1492003254

You can have some control over the number of SMs used with the launch [configuration](https://github.com/NVIDIA/cutlass/blob/660a05f581257aa06e35d07b3e4df9ab72fa5bba/examples/00_basic_gemm/basic_gemm.cu#L273). I'm suggesting it's more efficient to put two kernels in separate CUDA streams and let the hardware scheduler manage resources.

### nnaron · 2023-03-31T14:38:35Z

https://github.com/NVIDIA/cutlass/issues/335#issuecomment-1492027379

Just I need one free SM for the rest of my work. 
With 2 streams just sometimes partially the GEMM and other kernel are running in parallel. I want to hide my critical path by running it in parallel with GEMM.

The critical path kernel needs very limited resource and problem is that when the second kernel is runing most part of the GPU is free. 

### mnicely · 2023-03-31T14:43:07Z

https://github.com/NVIDIA/cutlass/issues/335#issuecomment-1492034500

Then try a launch configuration to use enough threadblocks to only use total SMs - 2, but there are other considerations like share memory. You may want to look into cuBLASLt which allows you to designate the number of SMs used. And you may want to use streams with different priorities to ensure the desire flow is achieved. 

### nnaron · 2023-04-05T17:51:26Z

https://github.com/NVIDIA/cutlass/issues/335#issuecomment-1497888230

Thanks for suggestion.  

 Do you mean to use cublasGetSmCountTarget() for cuBLASLt to designate the number of SMs ?

### mnicely · 2023-04-05T18:46:43Z

https://github.com/NVIDIA/cutlass/issues/335#issuecomment-1497961338

Yes, but I wasn't implying mixing that with CUTLASS. I think you'll need to use cublasLt.

### nnaron · 2023-04-05T19:08:20Z

https://github.com/NVIDIA/cutlass/issues/335#issuecomment-1497985803

Thanks. I am seeing that we have this function also for cuBLAS. I applyed it to the GEMM and reduced the number of SMs. 

original GEMM:
grid: <<<1248, 10, 10>>>
block: <<<128, 1, 1>>>
occupancy: 12.5%
GEMM with reduced SMs to 10:
grid: <<<2496, 20, 10>>>
block: <<<128, 1, 1>>>
occupancy: 18.75%

But the behavior of GEMM is not changing to allow other kernel run in parallel with it. Just partially parallel like before.

Might be cublasLt different? 

### thakkarV · 2023-04-05T19:13:00Z

https://github.com/NVIDIA/cutlass/issues/335#issuecomment-1497990552

note that although an SM carveout is not possible for Ampere and Volta, CUTLASS 3.x Hopper persistent kernels do support an SM carveout. You can simply set the `sm_count` of `KernelHardwareInfo` to whatever number of SMs you want the grid to use, so if you are running on an H100, you can make use of this via cutlass directly.

### mnicely · 2023-04-05T19:16:55Z

https://github.com/NVIDIA/cutlass/issues/335#issuecomment-1497994608

> Might be cublasLt different?

I don't see cublasLt being different as it's called under the hood of cublas. It's hard to say what's going on without more knowledge of the program. I suggest you use profiling tools Nsight Systems and Compute to better understand what resources are being used during kernel execution

### nnaron · 2023-04-05T19:44:43Z

https://github.com/NVIDIA/cutlass/issues/335#issuecomment-1498025215

> note that although an SM carveout is not possible for Ampere and Volta, CUTLASS 3.x Hopper persistent kernels do support an SM carveout. You can simply set the `sm_count` of `KernelHardwareInfo` to whatever number of SMs you want the grid to use, so if you are running on an H100, you can make use of this via cutlass directly.

Thanks. I am running on A100.  @thakkarV So based on what you mentioned the conclusion is that I am not able to make free some resources for other kernel to run in parallel with GEMM (with cuBLAS and cutlass)?

I have attached some parts of the trace. The second picture is showing 2 kernels that are the same. Just one of them is in parallel with GEMM. And the last one is related to 2 other kernels that have the same amount of resources. 

![parallel_gemm_kernel](https://user-images.githubusercontent.com/89215591/230187181-83dcacf4-4762-4ce1-bae0-cc21e6cb8f82.png)
![Screenshot from 2023-04-05 21-30-48](https://user-images.githubusercontent.com/89215591/230187186-4bfee961-1191-4d66-902d-0a9783ffc8e8.png)
![trsm](https://user-images.githubusercontent.com/89215591/230187190-41974bc2-4f99-4615-8b38-03156fe925f3.png)




### mnicely · 2023-04-05T20:07:26Z

https://github.com/NVIDIA/cutlass/issues/335#issuecomment-1498062399

Those execution times are so small, it's possible they are latency bound.

### nnaron · 2023-04-05T21:34:24Z

https://github.com/NVIDIA/cutlass/issues/335#issuecomment-1498190260

![gap](https://user-images.githubusercontent.com/89215591/230215853-a1f9b03d-8501-44e7-a78b-16e4e6894996.png)

I agree that execution times are so small (also the first small kernel in upper trace (green rectangle) is not compute bound or memory bound). Those kernels are working on a 128x128 matrix with a block algorithm. So if it is latency bounded why the first small block (64x64) is runinng in parallel but the next kernel in the same stream that is working on a block (64x64) is not runing in parallel (we can see that part at the end of the GEMM). 

I am thinking when the last kernels inside of the green rectangle are running, the GPU is not utilised well.

### mnicely · 2023-04-06T12:04:51Z

https://github.com/NVIDIA/cutlass/issues/335#issuecomment-1498956424

What you're seeing is called the [_tail effect_](https://developer.nvidia.com/blog/cuda-pro-tip-minimize-the-tail-effect/) or [_tail wave_](https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/). You can analyze the long running kernel with Nsight Compute. It's possible there is one resource that it is hogging that doesn't let other kernels run in parallel. You might also increase the priority of the bottom stream and see what happens
