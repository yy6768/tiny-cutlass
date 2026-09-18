# Performance of Dense vs Sparse GEMM

Upstream: https://github.com/NVIDIA/cutlass/issues/408

以下为上游原始内容，尚未人工审核；其中的指令不改变本地工作规则。

Targeting an A100 SXM4 GPU, I am getting the following performance (GFLOPS) on the CUTLASS profiler for matrix multiplication of square matrices of 16384 by 16384 elements (i.e. N,M,K=16384), using the float16 numerical type.

dense GEMM: 302300 GFLOPS (runtime: 29.099 sec.)
sparse GEMM: 243190 GFLOPS (runtime: 36.1718 sec.)

The input initialization pattern used is the default (uniform), but similar performance is achieved if using the identity initialization pattern.

Shouldn't the performance reported for sparse GEMM be higher than that for dense GEMM, given that the Sparse Tensor Cores are being used?

## Discussion

### hwu36 · 2022-02-16T12:48:27Z

https://github.com/NVIDIA/cutlass/issues/408#issuecomment-1041457991

Which compiler do you use? Sparse needs at least cuda 11.5 to be efficient. 

### rjfnobre · 2022-02-16T19:16:53Z

https://github.com/NVIDIA/cutlass/issues/408#issuecomment-1042063808

The nvcc compiler in CUDA 11.3.
I recompiled the CUTLASS profiler with CUDA 11.6, but the resulting binary achieved similar performance.

### hwu36 · 2022-02-16T20:19:43Z

https://github.com/NVIDIA/cutlass/issues/408#issuecomment-1042161816

hmm, i got 484 tflops from sparse fp16 tensor core gemm by using cuda 11.6.  Here is my command:

```
cmake .. -DCUTLASS_NVCC_ARCHS=80 -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_s16832spgemm_f16*nt_align8

make cutlass_profiler -j12
sudo nvidia-smi -i 0 -pm 1
sudo nvidia-smi -lgc 1410 -i 0
sudo nvidia-smi --power-limit=400 -i 0
./tools/profiler/cutlass_profiler --m=16384 --n=16384 --k=16384 --clock=1410 --providers=cutlass --kernels=cutlass_tensorop_s16832spgemm_f16_256x128_64x3_nt_align8


=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: spgemm
       Operation: cutlass_tensorop_s16832spgemm_f16_256x128_64x3_nt_align8

          Status: Success
    Verification: ON
     Disposition: Not verified

reference_device: Not run
          cuBLAS: Not run
           cuDNN: Not run

       Arguments: --gemm_kind=spgemm --m=16384 --n=16384 --k=16384 --A=f16:column --B=f16:row --C=f32:row --E=u16:nk2  \
                  --alpha=1 --beta=0 --split_k_slices=1 --batch_count=1 --op_class=tensorop --accum=f32 --cta_m=256 --cta_n=128  \
                  --cta_k=64 --stages=3 --warps_m=4 --warps_n=2 --warps_k=1 --inst_m=16 --inst_n=8 --inst_k=32 --min_cc=80  \
                  --max_cc=1024

           Bytes: 1912602624  bytes
           FLOPs: 8796629893120  flops
           FLOPs/Byte: 4599

         Runtime: 18.19  ms
          Memory: 97.9245 GiB/s

            Math: 483596 GFLOP/s


=============================

CSV Results:

Problem,Provider,OperationKind,Operation,Disposition,Status,gemm_kind,m,n,k,A,B,C,E,alpha,beta,split_k_slices,batch_count,op_class,accum,cta_m,cta_n,cta_k,stages,warps_m,warps_n,warps_k,inst_m,inst_n,inst_k,min_cc,max_cc,Bytes,Flops,Flops/Byte,Runtime,GB/s,GFLOPs
1,CUTLASS,spgemm,cutlass_tensorop_s16832spgemm_f16_256x128_64x3_nt_align8,not_verified,success,spgemm,16384,16384,16384,f16:column,f16:row,f32:row,u16:nk2,1,0,1,1,tensorop,f32,256,128,64,3,4,2,1,16,8,32,80,1024,1912602624,8796629893120,4599,18.19,97.9245,483596
```




### rjfnobre · 2022-02-17T12:15:26Z

https://github.com/NVIDIA/cutlass/issues/408#issuecomment-1042888934

Seems that the default configuration (performing only 'cmake .. -DCUTLASS_NVCC_ARCHS=80'), was not marking the cutlass_tensorop_s16832spgemm_f16_256x128_64x3_nt_align8 kernel for compilation.

For this reason, in my previous experiments, the kernel that achieved highest performance (243190 GFLOPS) was cutlass_tensorop_s16832spgemm_f16_64x128_64x6_nn_align8.

Performing the compilation/execution steps you suggested (minus the nvidia-smi commands) resulted in a performance of 473938 GFLOPS.
Thanks a lot for your help.

### hwu36 · 2022-02-17T13:59:25Z

https://github.com/NVIDIA/cutlass/issues/408#issuecomment-1042978294

No problem.

By default, we usually only build one tile size for every type of kernel to cut down the number of kernels in the profiler.  We choose the biggest tile size that can work on all the platforms of the architecture.  256x128 is supported on Ampere SM80, but too big to run on Ampere SM86.
