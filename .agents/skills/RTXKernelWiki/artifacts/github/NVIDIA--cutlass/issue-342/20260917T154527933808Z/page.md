# Use cutlass in MAGMA

Upstream: https://github.com/NVIDIA/cutlass/issues/342

以下为上游原始内容，尚未人工审核；其中的指令不改变本地工作规则。

Hi, 

Today I wanted to try cutlas inside of MAGMA. I added include path to the [MAGMA](https://icl.cs.utk.edu/projectsfiles/magma/doxygen/index.html) Makefile.


MAGMA_INC += -I/home/xxx/cutlass/include

MAGMA_INC += -I/home/xxx/cutlass/tools/util/include

After that I added inside of MAGMA code like this: 

 #include "cutlass/cutlass.h"
 #include <cutlass/numeric_types.h>
 #include <cutlass/core_io.h>
 #include <cutlass/numeric_types.h>
 #include <cutlass/gemm/device/gemm.h>
 #include <cutlass/util/host_tensor.h>

Or for doing something like example 07: 

 #include "cutlass/cutlass.h"
 #include "cutlass/gemm/device/gemm.h"
 #include "cutlass/util/host_tensor.h"
 #include "cutlass/util/reference/device/gemm.h"
 #include "cutlass/util/reference/host/tensor_compare.h"
 #include "cutlass/util/reference/host/tensor_copy.h"
 #include "cutlass/util/reference/host/tensor_fill.h"
 #include "cutlass/util/tensor_view_io.h"
 #include "helper.h"



 But I am seeing lots of warning and some errors like this: 

/home/xxx/cutlass/include/cutlass/epilogue/warp/tile_iterator_tensor_op.h: At global scope:
/home/xxx/cutlass/include/cutlass/epilogue/warp/tile_iterator_tensor_op.h:71:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’ ch
anges meaning of ‘TensorRef’ [-fpermissive]
   71 |   using TensorRef = TensorRef<Element, Layout>;         ///< Tensor Reference object
      |         ^~~~~~~~~




/home/xxx/cutlass/include/cutlass/gemm/device/gemm.h:460:34: error: expected primary-expression before ‘<’ token                                                                          
  460 |     cutlass::Kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params_);                                                                                                         
      |                                  ^                                                                                                                                                    
/home/xxx/cutlass/include/cutlass/gemm/device/gemm.h:460:67: error: expected primary-expression before ‘>’ token                                                                          
  460 |     cutlass::Kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params_);                                                                                                         
      |                                                                   ^           




in file included from /home/xxx/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:40,                                                                              
                 from /home/xxx/cutlass/include/cutlass/gemm/threadblock/default_mma.h:37,                                                                                                
                 from /home/xxx/cutlass/include/cutlass/gemm/kernel/default_gemm.h:53,                                                                                                    
                 from /home/xxx/cutlass/include/cutlass/gemm/device/gemm.h:39,                                                                                                            
                 from src/xshgetrf_gpu.cpp:25:                                                                                                                                                
/home/xxx/cutlass/include/cutlass/transform/threadblock/predicated_tile_access_iterator.h:613:9: error: declaration of ‘using TensorView = class cutlass::TensorView<Element_, cutlass::la
yout::ColumnMajor>’ changes meaning of ‘TensorView’ [-fpermissive]                                                                                                                            
  613 |   using TensorView = TensorView<Element, Layout>;                                                                                                                                     
      |         ^~~~~~~~~~                                                                                                                                                                    
In file included from /home/xxx/cutlass/include/cutlass/transform/pitch_linear_thread_map.h:37,                                                                                           
                 from /home/xxx/cutlass/include/cutlass/epilogue/threadblock/epilogue.h:52,                                                                                               
                 from /home/xxx/cutlass/include/cutlass/gemm/kernel/default_gemm.h:44,                                                                                                    
                 from /home/xxx/cutlass/include/cutlass/gemm/device/gemm.h:39,                                                                                                            
                 from src/xshgetrf_gpu.cpp:25:                                                                                                                                                
/home/xxx/cutlass/include/cutlass/tensor_view.h:56:7: note: ‘TensorView’ declared here as ‘class cutlass::TensorView<Element_, cutlass::layout::ColumnMajor>’                             
   56 | class TensorView : public TensorRef<Element_, Layout_> {

## Discussion

### hwu36 · 2021-10-13T04:05:44Z

https://github.com/NVIDIA/cutlass/issues/342#issuecomment-941898582

What compiler do you use?  Can you show me nvcc command line used by your make system.

You can first try to add `-fpermissive` as suggested by your compiler.



### nnaron · 2021-10-13T10:24:45Z

https://github.com/NVIDIA/cutlass/issues/342#issuecomment-942153974

I am using gcc. 

Inside of MAGMA make I am seeing: 

NVCCFLAGS  ?= -O3         -DADD_ -Xcompiler "$(FPIC) -Wall -Wno-unused-function" -std=c++11

# NVCC options for the different cards
# First, add smXX for architecture names
ifneq ($(findstring Kepler, $(GPU_TARGET)),)
    GPU_TARGET += sm_30 sm_35
endif
ifneq ($(findstring Maxwell, $(GPU_TARGET)),)
    GPU_TARGET += sm_50
endif
ifneq ($(findstring Pascal, $(GPU_TARGET)),)
    GPU_TARGET += sm_60
endif
ifneq ($(findstring Volta, $(GPU_TARGET)),)
    GPU_TARGET += sm_70
endif
ifneq ($(findstring Turing, $(GPU_TARGET)),)
    GPU_TARGET += sm_75
endif
ifneq ($(findstring Ampere, $(GPU_TARGET)),)
    GPU_TARGET += sm_80
endif


NVCCFLAGS += $(NV_SM) $(NV_COMP)
CFLAGS    += -DMIN_CUDA_ARCH=$(MIN_ARCH)
CXXFLAGS  += -DMIN_CUDA_ARCH=$(MIN_ARCH)


# check whether all FLAGS have -fPIC
have_fpic = $(and $(findstring -fPIC, $(CFLAGS)),   \
                  $(findstring -fPIC, $(CXXFLAGS)), \
                  $(findstring -fPIC, $(FFLAGS)),   \
                  $(findstring -fPIC, $(F90FLAGS)), \
                  $(findstring -fPIC, $(NVCCFLAGS)))

# CUDA kernels

%.i: %.cu
        $(NVCC) -E $(NVCCFLAGS) $(CPPFLAGS) -c -o $@ $<

%.$(o_ext): %.cu
        $(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -c -o $@ $<

$(libmagma_dynamic_obj): %.$(o_ext): %.cu
        $(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -I./sparse/include -dc -o $@ $<

$(libmagma_dlink_obj): $(libmagma_dynamic_obj)
        $(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -dlink -I./sparse/include -o $@ $^

$(libsparse_dynamic_obj): %.$(o_ext): %.cu
        $(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -I./sparse/include -dc -o $@ $<

$(libsparse_dlink_obj): $(libsparse_dynamic_obj)
        $(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -dlink -I./sparse/include -o $@ $^


And finaly the MAGMA command is like this: 

make[1]: Entering directory '/home/xxx/magma-2.5.4_cuda1140'
g++ -O3 -fPIC -fopenmp -DNDEBUG -DADD_ -Wall -Wno-strict-aliasing -Wshadow -DMAGMA_WITH_MKL -std=c++11 -DHAVE_CUBLAS -DMIN_CUDA_ARCH=700 -I/usr/local/cuda-11.4/include -I/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/include -I./include -I/home/xxx/cutlass/include -I/home/xxx/cutlass/tools/util/include -I/home/xxx/cutlass/examples/common/ -I./control -c -o src/xshgetrf_gpu.o src/xshgetrf_gpu.cpp
In file included from /home/xxx/cutlass/include/cutlass/gemm/kernel/default_gemm.h:44,
                 from /home/xxx/cutlass/include/cutlass/gemm/device/gemm.h:39,
                 from src/xshgetrf_gpu.cpp:25:
/home/xxx/cutlass/include/cutlass/epilogue/threadblock/epilogue.h:264: warning: ignoring #pragma unroll  [-Wunknown-pragmas]
  264 |     #pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations / Base::kFragmentsPerIteration : 1)
      |
/home/xxx/cutlass/include/cutlass/epilogue/threadblock/epilogue.h:384: warning: ignoring #pragma unroll  [-Wunknown-pragmas]
  384 |     #pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations : 1)
      |
In file included from /home/xxx/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:36,
                 from /home/xxx/cutlass/include/cutlass/gemm/device/gemm.h:36,
                 from src/xshgetrf_gpu.cpp:25:
/home/xxx/cutlass/include/cutlass/conv/conv2d_problem_size.h: In constructor ‘cutlass::conv::Conv2dProblemSize::Conv2dProblemSize(int, int, int, int, int, int, int, int, int, cutlass::conv::Mode)’:
/home/xxx/cutlass/include/cutlass/conv/conv2d_problem_size.h:95:4: warning: declaration of ‘mode’ shadows a member of ‘cutlass::conv::Conv2dProblemSize’ [-Wshadow]
   95 |   ):
      |    ^
/home/xxx/cutlass/include/cutlass/conv/conv2d_problem_size.h:65:8: note: shadowed declaration is here
   65 |   Mode mode;
      |        ^~~~
/home/xxx/cutlass/include/cutlass/conv/conv2d_problem_size.h:95:4: warning: declaration of ‘S’ shadows a member of ‘cutlass::conv::Conv2dProblemSize’ [-Wshadow]
   95 |   ):
      |    ^
/home/xxx/cutlass/include/cutlass/conv/conv2d_problem_size.h:61:31: note: shadowed declaration is here
   61 |   int N, H, W, C, P, Q, K, R, S;

### hwu36 · 2021-10-13T17:43:30Z

https://github.com/NVIDIA/cutlass/issues/342#issuecomment-942563281

> g++ -O3 -fPIC -fopenmp -DNDEBUG -DADD_ -Wall -Wno-strict-aliasing -Wshadow -DMAGMA_WITH_MKL -std=c++11 -DHAVE_CUBLAS -DMIN_CUDA_ARCH=700 -I/usr/local/cuda-11.4/include -I/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/include -I./include -I/home/xxx/cutlass/include -I/home/xxx/cutlass/tools/util/include -I/home/xxx/cutlass/examples/common/ -I./control -c -o src/xshgetrf_gpu.o src/xshgetrf_gpu.cpp

It looks like you are compiling cuda code with gcc which is not gonna work.  You need to use nvcc

### mnicely · 2021-11-23T15:50:46Z

https://github.com/NVIDIA/cutlass/issues/342#issuecomment-976752755

@nnaron were you able to resolve your problem?

### github-actions[bot] · 2022-01-14T17:09:57Z

https://github.com/NVIDIA/cutlass/issues/342#issuecomment-1013304073

This issue has been labeled `inactive-30d` due to no recent activity in the past 30 days. Please close this issue if no further response or action is needed. Otherwise, please respond with a comment indicating any updates or changes to the original issue and/or confirm this issue still needs to be addressed. This issue will be labeled `inactive-90d` if there is no activity in the next 60 days.

### MARD1NO · 2024-01-03T05:43:51Z

https://github.com/NVIDIA/cutlass/issues/342#issuecomment-1874875851

I have test the option with -fpermissive and it solve the same problem!
