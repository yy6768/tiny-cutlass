# Error in Build of CUTLASS Unit Tests

Upstream: https://github.com/NVIDIA/cutlass/issues/339

以下为上游原始内容，尚未人工审核；其中的指令不改变本地工作规则。

As I read the quick start guide, all unit test build should be successful. Is it right? 
I think all step before Unit Tests build was successful but now I am seeing this kind of error:

> make[3]: *** [test/unit/conv/device/CMakeFiles/cutlass_test_unit_conv_device_tensorop_f16_sm80.dir/build.make:76: test/unit/conv/device/CMakeFiles/cutlass_test_unit_conv_device_tensorop_f16_sm80.dir/conv2d_dgrad_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f16_sm80.cu.o] Error 137
> make[3]: *** Waiting for unfinished jobs....
> make[3]: *** [test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_sparse_tensorop_sm80.dir/build.make:206: test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_sparse_tensorop_sm80.dir/gemm_s8t_s8n_s32t_tensor_op_s32_sparse_sm80.cu.o] Error 137
> Killed
> make[3]: *** [test/unit/gemm/threadblock/CMakeFiles/cutlass_test_unit_gemm_threadblock.dir/build.make:63: test/unit/gemm/threadblock/CMakeFiles/cutlass_test_unit_gemm_threadblock.dir/mma_multistage.cu.o] Error 137
> 
> 
> 
> [ 94%] Built target cutlass_test_unit_gemm_device_tensorop_f16_sm80
> make[2]: *** [CMakeFiles/Makefile2:4942: test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_gemm_with_fused_epilogue_tensorop.dir/all] Error 2
> [ 94%] Linking CXX executable cutlass_test_unit_gemm_thread
> [ 94%] Built target cutlass_test_unit_gemm_thread
> make[2]: *** [CMakeFiles/Makefile2:5449: test/unit/conv/device/CMakeFiles/cutlass_test_unit_conv_device_tensorop_s32.dir/all] Error 2
> [ 94%] Linking CXX executable cutlass_test_unit_transform_threadblock
> [ 94%] Linking CXX executable cutlass_test_unit_conv_device_tensorop_f32_tf32_sm80
> [ 94%] Built target cutlass_test_unit_transform_threadblock
> [ 94%] Built target cutlass_test_unit_conv_device_tensorop_f32_tf32_sm80
> make[2]: *** [CMakeFiles/Makefile2:6457: test/unit/reduction/device/CMakeFiles/cutlass_test_unit_reduction_device.dir/all] Error 2
> 

The machine include 3 GPUs (A100, V100 & V100S). Also I did not specify the architecture during make .

I tryed to run some example to learn how cutlass is working, but it seems that I was not doing it in correct way I have access to A100 GPU. 

> xxx@xxx:~/cutlass/build/examples/14_ampere_tf32_tensorop_gemm$ make
> [100%] Built target 14_ampere_tf32_tensorop_gemm                
> xxx@xxx:~/cutlass/build/examples/14_ampere_tf32_tensorop_gemm$ ./14_ampere_tf32_tensorop_gemm 
> Ampere Tensor Core operations must be run on a machine with compute capability at least 80.
> xxx@xxx:~/cutlass/build/examples/14_ampere_tf32_tensorop_gemm$

Aha! Link: https://nvaiinfa.aha.io/features/CUTLASS-5

## Discussion

### hwu36 · 2021-10-08T21:20:03Z

https://github.com/NVIDIA/cutlass/issues/339#issuecomment-939122153

Would you paste the first error together with your cmake and make commandline?

As to your second question, you can use `export CUDA_VISIBLE_DEVICES=x` to choose which gpu to be your default gpu.

### nnaron · 2021-10-08T22:50:13Z

https://github.com/NVIDIA/cutlass/issues/339#issuecomment-939154994

mkdir build && cd build
cmake ..
make cutlass_profiler -j12                                                                                                                                     

> [ 94%] Built target cutlass_library_objs                                                                                                                                                      
> [ 94%] Built target cutlass_lib                                                                                                                                                               
> [100%] Built target cutlass_profiler   

Maybe this part: 

> [ 89%] Building CUDA object test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_wmma.dir/gemm_s4t_s4n_s32n_wmma_tensor_op_s32_sm75.cu.o                                            
> [ 91%] Building CUDA object test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_simt.dir/gemm_splitk_simt_sm50.cu.o                                                                
> [ 91%] Building CUDA object test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_wmma.dir/gemm_f16t_f16n_f16n_singlestage_wmma_tensor_op_f16_sm70.cu.o                              
> [ 91%] Building CUDA object test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_wmma.dir/gemm_f16t_f16n_f32t_singlestage_wmma_tensor_op_f32_sm70.cu.o                              
> Killed                                                                                                                                                                                        
> Killed                                                                                                                                                                                        
> Killed                                                                                                                                                                                        
> make[3]: *** [test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_sparse_tensorop_sm80.dir/build.make:115: test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_sparse_te
> nsorop_sm80.dir/gemm_f16t_f16n_f16t_tensor_op_f16_sparse_sm80.cu.o] Error 137                                                                                                                 
> Killed                                                                                                                                                                                        
> Killed                                                                                                                                                                                        
> Killed                                                                                                                                                                                        
> make[3]: *** Waiting for unfinished jobs....                                                                                                                                                  
> Killed                                                                                                                                                                                        
> Killed                                                                                                                                                                                        
> Killed                                                                                                                                                                                        
> Killed                                                                                                                                                                                        
> Killed                                                                                                                                                                                        
> make[3]: *** [test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm70.dir/build.make:115: test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm70.di
> r/gemm_f16n_f16n_f16t_volta_tensor_op_f32_sm70.cu.o] Error 137                                                                                                                                
> make[3]: *** Waiting for unfinished jobs....                                                                                                                                                  
> make[3]: *** [test/unit/conv/device/CMakeFiles/cutlass_test_unit_conv_device_tensorop_s32.dir/build.make:89: test/unit/conv/device/CMakeFiles/cutlass_test_unit_conv_device_tensorop_s32.dir/c
> onv2d_fprop_implicit_gemm_s8nhwc_s8nhwc_s32nhwc_tensor_op_s32_sm80.cu.o] Error 137                                                                                                            
> make[3]: *** [test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_f32_tf32_sm80.dir/build.make:102: test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensoro
> p_f32_tf32_sm80.dir/gemm_tf32t_tf32t_f32t_tensor_op_f32_sm80.cu.o] Error 137                                                                                                                  
> make[3]: *** Waiting for unfinished jobs....                                                                                                                                                  
> make[3]: *** Waiting for unfinished jobs....                                                                                                                                                  
> Killed                                                                                                                                                                                        
> c++: fatal error: Killed signal terminated program cc1plus                                                                                                                                    
> compilation terminated.                                                                                                                                                                       
> make[3]: *** [test/unit/conv/device/CMakeFiles/cutlass_test_unit_conv_device_tensorop_f32_sm80.dir/build.make:89: test/unit/conv/device/CMakeFiles/cutlass_test_unit_conv_device_tensorop_f32_
> sm80.dir/conv2d_wgrad_implicit_gemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32_sm80.cu.o] Error 137
> make[3]: *** Waiting for unfinished jobs....
> make[3]: *** [test/unit/nvrtc/CMakeFiles/cutlass_nvrtc.dir/build.make:1263: test/unit/nvrtc/CMakeFiles/cutlass_nvrtc.dir/cutlass/nvrtc/environment.cpp.o] Error 1
> make[3]: *** [test/unit/conv/device/CMakeFiles/cutlass_test_unit_conv_device_tensorop_s32_interleaved.dir/build.make:89: test/unit/conv/device/CMakeFiles/cutlass_test_unit_conv_device_tensor
> op_s32_interleaved.dir/conv2d_fprop_implicit_gemm_s8ncxhwx_s8cxrskx_s8ncxhwx_tensor_op_s32_sm80.cu.o] Error 137
> make[3]: *** Waiting for unfinished jobs....
> Killed
> Killed
> Killed
> 

### hwu36 · 2021-10-08T23:17:56Z

https://github.com/NVIDIA/cutlass/issues/339#issuecomment-939162814

Error 137 is due to out of memory.  https://support.circleci.com/hc/en-us/articles/115014359648-Exit-code-137-Out-of-memory

Try smaller number after -j

### dcerisano · 2022-04-20T01:18:02Z

https://github.com/NVIDIA/cutlass/issues/339#issuecomment-1103327312

Just a bump that 64GB of memory seems to be the min spec for `make test_unit -j` to run without OOMs on the first build.
Otherwise use some value for `j` that your system can handle (eg. half max threads).
Actually a very good stress test for a new rig. Crashed my new TR4 np. Filled the dimms and it flew through this.

### d-k-b · 2022-04-20T01:23:41Z

https://github.com/NVIDIA/cutlass/issues/339#issuecomment-1103333117

Yes, compiling the templated kernels can take a significant amount of memory. I would suggest a two step process similar to the following to use as many cores as possible to make faster progress, but recover if you run out of memory.

```
make cutlass_test_unit -j16 -k || make cutlass_test_unit -j8
make test_unit -j8
```
