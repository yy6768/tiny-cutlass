# Fail on Fused back-to-back INT8 NT interleaved GEMMs

Upstream: https://github.com/NVIDIA/cutlass/issues/221

以下为上游原始内容，尚未人工审核；其中的指令不改变本地工作规则。

> Running on SM80
> Running Non-fused back-to-back FP16 TN GEMMs...
> gemm 0 time 0.772536 ms
> gemm 1 time 0.423127 ms
> total time 1.19566 ms
> Pass
> Running Fused back-to-back FP16 TN GEMMs...
> time 1.08622 ms
> Pass
> Running Non-fused back-to-back INT8 NT interleaved GEMMs...
> gemm 0 time 0.423936 ms
> gemm 1 time 0.135834 ms
> total time 0.55977 ms
> Pass
> Running Fused back-to-back INT8 NT interleaved GEMMs...
> time 0.53076 ms
> ...contrib\cutlass-2.5.0\examples\13_two_tensor_op_fusion\b2b_interleaved_gemm_run.h 611: CHECK_GT failed
> ...contrib\cutlass-2.5.0\examples\13_two_tensor_op_fusion\b2b_interleaved_gemm_run.h 618: CHECK_TRUE failed
> Dumping results in error_B2bGemm_device_interleaved_fused.txt
> Fail

Windows 10, cuda 11.2, RTX 3090.

Is it a bug?

Aha! Link: https://nvaiinfa.aha.io/features/CUTLASS-41

## Discussion

### ocwins · 2021-04-02T22:58:40Z

https://github.com/NVIDIA/cutlass/issues/221#issuecomment-812746991

The dump file is too large to upload.

### hwu36 · 2021-04-02T23:50:57Z

https://github.com/NVIDIA/cutlass/issues/221#issuecomment-812757680

That is strange.   Would you please to show us how you configured and built.  I mainly want to know if you build to sm80 or sm86.

### ocwins · 2021-04-03T01:43:26Z

https://github.com/NVIDIA/cutlass/issues/221#issuecomment-812774389

> 
> 
> That is strange. Would you please to show us how you configured and built. I mainly want to know if you build to sm80 or sm86.

Both sm80 and sm86 have this issue. Visual Studio 2019 .

>"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin\nvcc.exe" -gencode=arch=compute_80,code=\"sm_80,compute_80\" --use-local-env -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Tools\MSVC\14.28.29910\bin\HostX86\x64" -x cu   -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include"     --keep-dir x64\Release -use_fast_math -maxrregcount=0  --machine 64 --compile -cudart static    -D_UNICODE -DUNICODE -Xcompiler "/EHsc /W4 /nologo /O2 /Fdx64\Release\test.pdb /FS   /MD " -o x64\Release\n0play\cutlass_test.cu.obj "cutlass_test.cu"

in cutlass_test.cu:
```
#include "../../contrib/cutlass-2.5.0/examples/13_two_tensor_op_fusion/b2b_conv2d_fprop_implicit_gemm_s8ncxhwx_s8cxrskx_s8ncxhwx_tensor_op_s32_sm75.h"
#include "../../contrib/cutlass-2.5.0/examples/13_two_tensor_op_fusion/b2b_conv2d_fprop_implicit_gemm_s8ncxhwx_s8cxrskx_s8ncxhwx_tensor_op_s32_sm80.h"
#include "../../contrib/cutlass-2.5.0/examples/13_two_tensor_op_fusion/b2b_conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f16_sm75.h"
#include "../../contrib/cutlass-2.5.0/examples/13_two_tensor_op_fusion/b2b_conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f16_sm80.h"

int run_b2b() {

    cudaDeviceProp props;

    cudaError_t error = cudaGetDeviceProperties(&props, 0);
    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
        return -1;
        }

    if (!(props.major * 10 + props.minor >= 75)) {
        std::cerr << "Turing Tensor Ops must be run on a machine with compute capability at least 75."
            << std::endl;

        // Returning zero so this test passes on older Toolkits. Its actions are no-op.
        return 0;
        }

#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)
    std::cout << "Running on SM80" << std::endl;
    run_nonfused_conv2d_fprop_optimized_f16_sm80();
    run_fused_conv2d_fprop_optimized_f16_sm80();
    run_nonfused_conv2d_fprop_optimized_s8_sm80();
    run_fused_conv2d_fprop_optimized_s8_sm80();
#elif defined(CUTLASS_ARCH_MMA_SM75_SUPPORTED)
    std::cout << "Running on SM75" << std::endl;
    run_nonfused_conv2d_fprop_optimized_f16_sm75();
    run_fused_conv2d_fprop_optimized_f16_sm75();
    run_nonfused_conv2d_fprop_optimized_s8_sm75();
    run_fused_conv2d_fprop_optimized_s8_sm75();
#endif

    return 0;
    }
```
Just copied from fused_conv2d.cu (without main()).  I start with sm_86,compute_86 and change it to sm_80,compute_80 .

### ocwins · 2021-04-03T01:47:29Z

https://github.com/NVIDIA/cutlass/issues/221#issuecomment-812774844

both conv2d and gemm are tested. they all fail on Fused back-to-back INT8.

> Running on SM80
> Running Non-fused back-to-back FP16 Optimized Convolution Fprops...
> conv2d 0 time 0.957317 ms
> conv2d 1 time 0.348047 ms
> total time 1.30536 ms
> Pass
> Running Fused back-to-back FP16 Optimized Convolution Fprops...
> time 1.0918 ms
> Pass
> Running Non-fused back-to-back INT8 interleaved Optimized Convolution Fprops...
> conv2d 0 time 0.619807 ms
> conv2d 1 time 0.218532 ms
> total time 0.838339 ms
> Pass
> Running Fused back-to-back INT8 interleaved Optimized Convolution Fprops...
> time 0.652155 ms
> ...\contrib\cutlass-2.5.0\examples\13_two_tensor_op_fusion\b2b_interleaved_conv2d_run.h 622: CHECK_GT failed
> ...\contrib\cutlass-2.5.0\examples\13_two_tensor_op_fusion\b2b_interleaved_conv2d_run.h 629: CHECK_TRUE failed
> Dumping results in error_B2bImplicitGemm_device_interleaved_fused.txt
> Fail
> Press any key to continue . . .

### ocwins · 2021-04-03T02:12:19Z

https://github.com/NVIDIA/cutlass/issues/221#issuecomment-812777940

I have checked the dump file. There are all zeros under "D1 computed".

### ocwins · 2021-04-03T04:18:38Z

https://github.com/NVIDIA/cutlass/issues/221#issuecomment-812808003

Some tests on FP16 Optimized Convolution.

with:
```
enum { N = 128 };
enum { C = 256 };
enum { H = 10 };
enum { W = 9 };
cutlass::conv::Conv2dProblemSize conv2d_f16_sm80_problem_size_0(
    { N, H, W, C },    // input size (NHWC)
    { C, 3, 3, C },   // filter size (KRSC)
    { 1, 1, 1, 1 },     // padding (pad_h, _, pad_w, _)
    { 1, 1 },           // stride (stride_h, stride_w)
    { 1, 1 },           // dilation (dilation_h, dilation_w)
    { N, H, W, C }     // output size (NPQK)
);
cutlass::conv::Conv2dProblemSize conv2d_f16_sm80_problem_size_1(
    { N, H, W, C },    // input size (NHWC)
    { C, 3, 3, C },   // filter size (KRSC)                             <--------- changed
    { 1, 1, 1, 1 },     // padding (pad_h, _, pad_w, _)         <--------- changed
    { 1, 1 },           // stride (stride_h, stride_w)
    { 1, 1 },           // dilation (dilation_h, dilation_w)
    { N, H, W, C }     // output size (NPQK)
);
```
Non-fused and Fused versions both fails.

with:
```
enum { N = 128 };
enum { C = 256 };
enum { H = 10 };
enum { W = 9 };
cutlass::conv::Conv2dProblemSize conv2d_f16_sm80_problem_size_0(
    { N, H, W, C },    // input size (NHWC)
    { C, 3, 3, C },   // filter size (KRSC)                          
    { 1, 1, 1, 1 },     // padding (pad_h, _, pad_w, _)     
    { 1, 1 },           // stride (stride_h, stride_w)
    { 1, 1 },           // dilation (dilation_h, dilation_w)
    { N, H, W, C }     // output size (NPQK)
);
cutlass::conv::Conv2dProblemSize conv2d_f16_sm80_problem_size_1(
    { N, H, W, C },    // input size (NHWC)
    { C, 1, 1, C },   // filter size (KRSC)                       <--------- R, S is original
    { 0, 0, 0, 0 },     // padding (pad_h, _, pad_w, _)   <--------- R, S is original
    { 1, 1 },           // stride (stride_h, stride_w)
    { 1, 1 },           // dilation (dilation_h, dilation_w)
    { N, H, W, C }     // output size (NPQK)
);
```
Non-fused pass but Fused fails.

In the dump file, values under "D1 computed" are not zero , but they are different from "D1 reference"



### ocwins · 2021-04-03T04:31:26Z

https://github.com/NVIDIA/cutlass/issues/221#issuecomment-812809267

with:
```
enum { N = 128 };
enum { C = 64 };     //<----------------- it seems C can't be greater than 64 with fused ?
enum { H = 10 };
enum { W = 9 };
cutlass::conv::Conv2dProblemSize conv2d_f16_sm80_problem_size_0(
    { N, H, W, C },    // input size (NHWC)
    { C, 3, 3, C },   // filter size (KRSC)
    { 1, 1, 1, 1 },     // padding (pad_h, _, pad_w, _)
    { 1, 1 },           // stride (stride_h, stride_w)
    { 1, 1 },           // dilation (dilation_h, dilation_w)
    { N, H, W, C }     // output size (NPQK)
);
cutlass::conv::Conv2dProblemSize conv2d_f16_sm80_problem_size_1(
    { N, H, W, C },    // input size (NHWC)
    { C, 1, 1, C },   // filter size (KRSC)
    { 0, 0, 0, 0 },     // padding (pad_h, _, pad_w, _)
    { 1, 1 },           // stride (stride_h, stride_w)
    { 1, 1 },           // dilation (dilation_h, dilation_w)
    { N, H, W, C }     // output size (NPQK)
);
```

Both pass.

but if we change filter size and padding of problem_1:
```
enum { N = 128 };
enum { C = 64 };
enum { H = 10 };
enum { W = 9 };
cutlass::conv::Conv2dProblemSize conv2d_f16_sm80_problem_size_0(
    { N, H, W, C },    // input size (NHWC)
    { C, 3, 3, C },   // filter size (KRSC)
    { 1, 1, 1, 1 },     // padding (pad_h, _, pad_w, _)
    { 1, 1 },           // stride (stride_h, stride_w)
    { 1, 1 },           // dilation (dilation_h, dilation_w)
    { N, H, W, C }     // output size (NPQK)
);
cutlass::conv::Conv2dProblemSize conv2d_f16_sm80_problem_size_1(
    { N, H, W, C },    // input size (NHWC)
    { C, 3, 3, C },   // filter size (KRSC)     <---------------- change here
    { 1, 1, 1, 1 },     // padding (pad_h, _, pad_w, _)  <---------------- and here
    { 1, 1 },           // stride (stride_h, stride_w)
    { 1, 1 },           // dilation (dilation_h, dilation_w)
    { N, H, W, C }     // output size (NPQK)
);
```

Non-fused passes and Fused fail. But as another test shows above, with same filter size and padding, but C was 256, they both failed.



### hwu36 · 2021-04-04T04:37:42Z

https://github.com/NVIDIA/cutlass/issues/221#issuecomment-812970176

We root caused issues.  There is a bug in the relu epilogue.  Thank you very much.  We can show you the needed change soon after we finish the testing.

### ocwins · 2021-04-04T09:13:46Z

https://github.com/NVIDIA/cutlass/issues/221#issuecomment-813000680

> 
> 
> We root caused issues. There is a bug in the relu epilogue. Thank you very much. We can show you the needed change soon after we finish the testing.

Really a good news to me :)

I have suspected the relu, because other examples without it run well. But I'm not familiar with cutlass at the moment, so I can't figure out where the problem actually is.

 It's amazing the issues can be rooted so quickly. So I can be back to my own work soon. Thank you!

P.S.  Maybe I shouldn't say, some posts on cuDNN forum took several months before it had been replied ;)


### hwu36 · 2021-04-06T15:27:02Z

https://github.com/NVIDIA/cutlass/issues/221#issuecomment-814212029

Alright, here is the fix.  Only the one in `linear_combination_relu.h` is the functional fix.  The rest is to make the example more robust.

```diff --git a/examples/13_two_tensor_op_fusion/b2b_conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f16_sm75.h b/examples/13_two_tensor_op_fusion/b2b_conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f16_sm75.h
index 305d182..53b2fc8 100644
--- a/examples/13_two_tensor_op_fusion/b2b_conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f16_sm75.h
+++ b/examples/13_two_tensor_op_fusion/b2b_conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f16_sm75.h
@@ -36,8 +36,6 @@
 #include "device/b2b_implicit_gemm_convolution.h"
 #include "b2b_conv2d_run.h"
 
-#if defined(CUTLASS_ARCH_MMA_SM75_SUPPORTED)
-
 ////////////////////////////////////////////////////////////////////////////////
 
 cutlass::conv::Conv2dProblemSize conv2d_f16_sm75_problem_size_0 (
@@ -57,7 +55,7 @@ cutlass::conv::Conv2dProblemSize conv2d_f16_sm75_problem_size_1 (
     {128, 56, 56, 64}     // output size (NPQK)
   );
 
-void run_nonfused_conv2d_fprop_f16_sm75() {
+bool run_nonfused_conv2d_fprop_f16_sm75() {
 
   using ElementA           = cutlass::half_t;
   using ElementB           = cutlass::half_t;
@@ -90,7 +88,8 @@ void run_nonfused_conv2d_fprop_f16_sm75() {
       ElementC,
       128 / cutlass::sizeof_bits<ElementC>::value,
       ElementAccumulator,
-      ElementCompute
+      ElementCompute,
+      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
     >,
     cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
     2,
@@ -135,9 +134,10 @@ void run_nonfused_conv2d_fprop_f16_sm75() {
   else
     std::cout << "Fail\n";
 
+  return pass;
 }
 
-void run_fused_conv2d_fprop_f16_sm75() {
+bool run_fused_conv2d_fprop_f16_sm75() {
 
   using ElementA           = cutlass::half_t; 
   using ElementB           = cutlass::half_t; 
@@ -161,7 +161,8 @@ void run_fused_conv2d_fprop_f16_sm75() {
       ElementC,
       InstructionShape::kM * InstructionShape::kN / 32,
       ElementAccumulator,
-      ElementCompute
+      ElementCompute,
+      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
     >;
 
   using EpilogueOutputOp1 = 
@@ -207,9 +208,10 @@ void run_fused_conv2d_fprop_f16_sm75() {
   else
     std::cout << "Fail\n";
 
+  return pass;
 }
 
-void run_nonfused_conv2d_fprop_optimized_f16_sm75() {
+bool run_nonfused_conv2d_fprop_optimized_f16_sm75() {
 
   using ElementA           = cutlass::half_t;
   using ElementB           = cutlass::half_t;
@@ -242,7 +244,8 @@ void run_nonfused_conv2d_fprop_optimized_f16_sm75() {
       ElementC,
       128 / cutlass::sizeof_bits<ElementC>::value,
       ElementAccumulator,
-      ElementCompute
+      ElementCompute,
+      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
     >,
     cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
     2,
@@ -287,9 +290,10 @@ void run_nonfused_conv2d_fprop_optimized_f16_sm75() {
   else
     std::cout << "Fail\n";
 
+  return pass;
 }
 
-void run_fused_conv2d_fprop_optimized_f16_sm75() {
+bool run_fused_conv2d_fprop_optimized_f16_sm75() {
 
   using ElementA           = cutlass::half_t; 
   using ElementB           = cutlass::half_t; 
@@ -313,7 +317,8 @@ void run_fused_conv2d_fprop_optimized_f16_sm75() {
       ElementC,
       InstructionShape::kM * InstructionShape::kN / 32,
       ElementAccumulator,
-      ElementCompute
+      ElementCompute,
+      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
     >;
 
   using EpilogueOutputOp1 = 
@@ -359,10 +364,8 @@ void run_fused_conv2d_fprop_optimized_f16_sm75() {
   else
     std::cout << "Fail\n";
 
+  return pass;
 }
 
 
 ////////////////////////////////////////////////////////////////////////////////
-
-#endif  // if defined(CUTLASS_ARCH_MMA_SM75_SUPPORTED)
-
diff --git a/examples/13_two_tensor_op_fusion/b2b_conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f16_sm80.h b/examples/13_two_tensor_op_fusion/b2b_conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f16_sm80.h
index e14134e..7451c5b 100644
--- a/examples/13_two_tensor_op_fusion/b2b_conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f16_sm80.h
+++ b/examples/13_two_tensor_op_fusion/b2b_conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f16_sm80.h
@@ -36,8 +36,6 @@
 #include "device/b2b_implicit_gemm_convolution.h"
 #include "b2b_conv2d_run.h"
 
-#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)
-
 ////////////////////////////////////////////////////////////////////////////////
 
 cutlass::conv::Conv2dProblemSize conv2d_f16_sm80_problem_size_0 (
@@ -57,7 +55,7 @@ cutlass::conv::Conv2dProblemSize conv2d_f16_sm80_problem_size_1 (
     {128, 56, 56, 64}     // output size (NPQK)
   );
 
-void run_nonfused_conv2d_fprop_f16_sm80() {
+bool run_nonfused_conv2d_fprop_f16_sm80() {
 
   using ElementA           = cutlass::half_t;
   using ElementB           = cutlass::half_t;
@@ -90,7 +88,8 @@ void run_nonfused_conv2d_fprop_f16_sm80() {
       ElementC,
       128 / cutlass::sizeof_bits<ElementC>::value,
       ElementAccumulator,
-      ElementCompute
+      ElementCompute,
+      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
     >,
     cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
     3,
@@ -135,9 +134,10 @@ void run_nonfused_conv2d_fprop_f16_sm80() {
   else
     std::cout << "Fail\n";
 
+  return pass;
 }
 
-void run_fused_conv2d_fprop_f16_sm80() {
+bool run_fused_conv2d_fprop_f16_sm80() {
 
   using ElementA           = cutlass::half_t; 
   using ElementB           = cutlass::half_t; 
@@ -161,7 +161,8 @@ void run_fused_conv2d_fprop_f16_sm80() {
       ElementC,
       InstructionShape::kM * InstructionShape::kN / 32,
       ElementAccumulator,
-      ElementCompute
+      ElementCompute,
+      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
     >;
 
   using EpilogueOutputOp1 = 
@@ -205,9 +206,10 @@ void run_fused_conv2d_fprop_f16_sm80() {
   else
     std::cout << "Fail\n";
 
+  return pass;
 }
 
-void run_nonfused_conv2d_fprop_optimized_f16_sm80() {
+bool run_nonfused_conv2d_fprop_optimized_f16_sm80() {
 
   using ElementA           = cutlass::half_t;
   using ElementB           = cutlass::half_t;
@@ -240,7 +242,8 @@ void run_nonfused_conv2d_fprop_optimized_f16_sm80() {
       ElementC,
       128 / cutlass::sizeof_bits<ElementC>::value,
       ElementAccumulator,
-      ElementCompute
+      ElementCompute,
+      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
     >,
     cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
     3,
@@ -285,9 +288,10 @@ void run_nonfused_conv2d_fprop_optimized_f16_sm80() {
   else
     std::cout << "Fail\n";
 
+  return pass;
 }
 
-void run_fused_conv2d_fprop_optimized_f16_sm80() {
+bool run_fused_conv2d_fprop_optimized_f16_sm80() {
 
   using ElementA           = cutlass::half_t; 
   using ElementB           = cutlass::half_t; 
@@ -311,7 +315,8 @@ void run_fused_conv2d_fprop_optimized_f16_sm80() {
       ElementC,
       InstructionShape::kM * InstructionShape::kN / 32,
       ElementAccumulator,
-      ElementCompute
+      ElementCompute,
+      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
     >;
 
   using EpilogueOutputOp1 = 
@@ -355,9 +360,8 @@ void run_fused_conv2d_fprop_optimized_f16_sm80() {
   else
     std::cout << "Fail\n";
 
+  return pass;
 }
 
 ////////////////////////////////////////////////////////////////////////////////
 
-#endif  // if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)
-
diff --git a/examples/13_two_tensor_op_fusion/b2b_conv2d_fprop_implicit_gemm_s8ncxhwx_s8cxrskx_s8ncxhwx_tensor_op_s32_sm75.h b/examples/13_two_tensor_op_fusion/b2b_conv2d_fprop_implicit_gemm_s8ncxhwx_s8cxrskx_s8ncxhwx_tensor_op_s32_sm75.h
index 2cb4ac2..c7ba4d9 100644
--- a/examples/13_two_tensor_op_fusion/b2b_conv2d_fprop_implicit_gemm_s8ncxhwx_s8cxrskx_s8ncxhwx_tensor_op_s32_sm75.h
+++ b/examples/13_two_tensor_op_fusion/b2b_conv2d_fprop_implicit_gemm_s8ncxhwx_s8cxrskx_s8ncxhwx_tensor_op_s32_sm75.h
@@ -36,8 +36,6 @@
 #include "device/b2b_implicit_gemm_convolution.h"
 #include "b2b_interleaved_conv2d_run.h"
 
-#if defined(CUTLASS_ARCH_MMA_SM75_SUPPORTED)
-
 ////////////////////////////////////////////////////////////////////////////////
 
 cutlass::conv::Conv2dProblemSize conv2d_s8_sm75_problem_size_0 (
@@ -57,7 +55,7 @@ cutlass::conv::Conv2dProblemSize conv2d_s8_sm75_problem_size_1 (
     {128, 56, 56, 64}     // output size (NPQK)
   );
 
-void run_nonfused_conv2d_fprop_s8_sm75() {
+bool run_nonfused_conv2d_fprop_s8_sm75() {
 
   using ElementA           = int8_t;
   using ElementB           = int8_t;
@@ -90,7 +88,8 @@ void run_nonfused_conv2d_fprop_s8_sm75() {
       ElementC,
       64 / cutlass::sizeof_bits<ElementC>::value,
       ElementAccumulator,
-      ElementCompute
+      ElementCompute,
+      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
     >,
     cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
     2,
@@ -135,9 +134,10 @@ void run_nonfused_conv2d_fprop_s8_sm75() {
   else
     std::cout << "Fail\n";
 
+  return pass;
 }
 
-void run_fused_conv2d_fprop_s8_sm75() {
+bool run_fused_conv2d_fprop_s8_sm75() {
 
   using ElementA           = int8_t;
   using ElementB           = int8_t;
@@ -161,7 +161,8 @@ void run_fused_conv2d_fprop_s8_sm75() {
       ElementC,
       InstructionShape::kM * InstructionShape::kN / 32,
       ElementAccumulator,
-      ElementCompute
+      ElementCompute,
+      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
     >;
 
   using EpilogueOutputOp1 = 
@@ -207,9 +208,10 @@ void run_fused_conv2d_fprop_s8_sm75() {
   else
     std::cout << "Fail\n";
 
+  return pass;
 }
 
-void run_nonfused_conv2d_fprop_optimized_s8_sm75() {
+bool run_nonfused_conv2d_fprop_optimized_s8_sm75() {
 
   using ElementA           = int8_t;
   using ElementB           = int8_t;
@@ -242,7 +244,8 @@ void run_nonfused_conv2d_fprop_optimized_s8_sm75() {
       ElementC,
       64 / cutlass::sizeof_bits<ElementC>::value,
       ElementAccumulator,
-      ElementCompute
+      ElementCompute,
+      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
     >,
     cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
     2,
@@ -287,9 +290,10 @@ void run_nonfused_conv2d_fprop_optimized_s8_sm75() {
   else
     std::cout << "Fail\n";
 
+  return pass;
 }
 
-void run_fused_conv2d_fprop_optimized_s8_sm75() {
+bool run_fused_conv2d_fprop_optimized_s8_sm75() {
 
   using ElementA           = int8_t;
   using ElementB           = int8_t;
@@ -313,7 +317,8 @@ void run_fused_conv2d_fprop_optimized_s8_sm75() {
       ElementC,
       InstructionShape::kM * InstructionShape::kN / 32,
       ElementAccumulator,
-      ElementCompute
+      ElementCompute,
+      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
     >;
 
   using EpilogueOutputOp1 = 
@@ -359,9 +364,8 @@ void run_fused_conv2d_fprop_optimized_s8_sm75() {
   else
     std::cout << "Fail\n";
 
+  return pass;
 }
 
 ////////////////////////////////////////////////////////////////////////////////
 
-#endif  // if defined(CUTLASS_ARCH_MMA_SM75_SUPPORTED)
-
diff --git a/examples/13_two_tensor_op_fusion/b2b_conv2d_fprop_implicit_gemm_s8ncxhwx_s8cxrskx_s8ncxhwx_tensor_op_s32_sm80.h b/examples/13_two_tensor_op_fusion/b2b_conv2d_fprop_implicit_gemm_s8ncxhwx_s8cxrskx_s8ncxhwx_tensor_op_s32_sm80.h
index c73d6c6..b1d9266 100644
--- a/examples/13_two_tensor_op_fusion/b2b_conv2d_fprop_implicit_gemm_s8ncxhwx_s8cxrskx_s8ncxhwx_tensor_op_s32_sm80.h
+++ b/examples/13_two_tensor_op_fusion/b2b_conv2d_fprop_implicit_gemm_s8ncxhwx_s8cxrskx_s8ncxhwx_tensor_op_s32_sm80.h
@@ -36,8 +36,6 @@
 #include "device/b2b_implicit_gemm_convolution.h"
 #include "b2b_interleaved_conv2d_run.h"
 
-#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)
-
 ////////////////////////////////////////////////////////////////////////////////
 
 cutlass::conv::Conv2dProblemSize conv2d_s8_sm80_problem_size_0 (
@@ -57,7 +55,7 @@ cutlass::conv::Conv2dProblemSize conv2d_s8_sm80_problem_size_1 (
     {128, 56, 56, 64}     // output size (NPQK)
   );
 
-void run_nonfused_conv2d_fprop_s8_sm80() {
+bool run_nonfused_conv2d_fprop_s8_sm80() {
 
   using ElementA           = int8_t;
   using ElementB           = int8_t;
@@ -90,7 +88,8 @@ void run_nonfused_conv2d_fprop_s8_sm80() {
       ElementC,
       64 / cutlass::sizeof_bits<ElementC>::value,
       ElementAccumulator,
-      ElementCompute
+      ElementCompute,
+      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
     >,
     cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
     3,
@@ -135,9 +134,10 @@ void run_nonfused_conv2d_fprop_s8_sm80() {
   else
     std::cout << "Fail\n";
 
+  return pass;
 }
 
-void run_fused_conv2d_fprop_s8_sm80() {
+bool run_fused_conv2d_fprop_s8_sm80() {
 
   using ElementA           = int8_t;
   using ElementB           = int8_t;
@@ -161,7 +161,8 @@ void run_fused_conv2d_fprop_s8_sm80() {
       ElementC,
       8 * InstructionShape::kN / 32,
       ElementAccumulator,
-      ElementCompute
+      ElementCompute,
+      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
     >;
 
   using EpilogueOutputOp1 = 
@@ -207,9 +208,10 @@ void run_fused_conv2d_fprop_s8_sm80() {
   else
     std::cout << "Fail\n";
 
+  return pass;
 }
 
-void run_nonfused_conv2d_fprop_optimized_s8_sm80() {
+bool run_nonfused_conv2d_fprop_optimized_s8_sm80() {
 
   using ElementA           = int8_t;
   using ElementB           = int8_t;
@@ -242,7 +244,8 @@ void run_nonfused_conv2d_fprop_optimized_s8_sm80() {
       ElementC,
       64 / cutlass::sizeof_bits<ElementC>::value,
       ElementAccumulator,
-      ElementCompute
+      ElementCompute,
+      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
     >,
     cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
     3,
@@ -287,9 +290,10 @@ void run_nonfused_conv2d_fprop_optimized_s8_sm80() {
   else
     std::cout << "Fail\n";
 
+  return pass;
 }
 
-void run_fused_conv2d_fprop_optimized_s8_sm80() {
+bool run_fused_conv2d_fprop_optimized_s8_sm80() {
 
   using ElementA           = int8_t;
   using ElementB           = int8_t;
@@ -313,7 +317,8 @@ void run_fused_conv2d_fprop_optimized_s8_sm80() {
       ElementC,
       8 * InstructionShape::kN / 32,
       ElementAccumulator,
-      ElementCompute
+      ElementCompute,
+      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
     >;
 
   using EpilogueOutputOp1 = 
@@ -359,10 +364,9 @@ void run_fused_conv2d_fprop_optimized_s8_sm80() {
   else
     std::cout << "Fail\n";
 
+  return pass;
 }
 
 
 ////////////////////////////////////////////////////////////////////////////////
 
-#endif  // if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)
-
diff --git a/examples/13_two_tensor_op_fusion/b2b_gemm_f16t_f16n_f16t_tensor_op_f16_sm75.h b/examples/13_two_tensor_op_fusion/b2b_gemm_f16t_f16n_f16t_tensor_op_f16_sm75.h
index 50da709..e0e2f45 100644
--- a/examples/13_two_tensor_op_fusion/b2b_gemm_f16t_f16n_f16t_tensor_op_f16_sm75.h
+++ b/examples/13_two_tensor_op_fusion/b2b_gemm_f16t_f16n_f16t_tensor_op_f16_sm75.h
@@ -39,14 +39,12 @@
 #include "device/b2b_gemm.h"
 #include "b2b_gemm_run.h"
 
-#if defined(CUTLASS_ARCH_MMA_SM75_SUPPORTED)
-
 ////////////////////////////////////////////////////////////////////////////////
 
 cutlass::gemm::GemmCoord gemm_f16_sm75_problem_size_0(128*1600, 64, 576);
 cutlass::gemm::GemmCoord gemm_f16_sm75_problem_size_1(128*1600, 128, 64);
 
-void run_nonfused_gemm_f16() {
+bool run_nonfused_gemm_f16() {
 
   using ElementOutput = cutlass::half_t;
   using ElementAccumulator = cutlass::half_t;
@@ -80,7 +78,8 @@ void run_nonfused_gemm_f16() {
       ElementOutput,
       128 / cutlass::sizeof_bits<ElementOutput>::value,
       ElementAccumulator,
-      ElementCompute
+      ElementCompute,
+      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
     >,
     cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
     2
@@ -116,9 +115,11 @@ void run_nonfused_gemm_f16() {
     std::cout << "Pass\n";
   else
     std::cout << "Fail\n";
+
+  return pass;
 }
 
-void run_fused_gemm_f16() {
+bool run_fused_gemm_f16() {
 
   using ElementOutput = cutlass::half_t;
   using ElementAccumulator = cutlass::half_t;
@@ -140,7 +141,8 @@ void run_fused_gemm_f16() {
       ElementOutput,
       InstructionShape::kM * InstructionShape::kN / 32,
       ElementAccumulator,
-      ElementCompute
+      ElementCompute,
+      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
     >;
 
   using EpilogueOutputOp1 = 
@@ -183,7 +185,6 @@ void run_fused_gemm_f16() {
   else
     std::cout << "Fail\n";
 
+  return passed;
 }
 ////////////////////////////////////////////////////////////////////////////////
-
-#endif  //#if defined(CUTLASS_ARCH_MMA_SM75_SUPPORTED)
diff --git a/examples/13_two_tensor_op_fusion/b2b_gemm_f16t_f16n_f16t_tensor_op_f16_sm80.h b/examples/13_two_tensor_op_fusion/b2b_gemm_f16t_f16n_f16t_tensor_op_f16_sm80.h
index 749ece2..3a64da8 100644
--- a/examples/13_two_tensor_op_fusion/b2b_gemm_f16t_f16n_f16t_tensor_op_f16_sm80.h
+++ b/examples/13_two_tensor_op_fusion/b2b_gemm_f16t_f16n_f16t_tensor_op_f16_sm80.h
@@ -39,14 +39,12 @@
 #include "device/b2b_gemm.h"
 #include "b2b_gemm_run.h"
 
-#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)
-
 ////////////////////////////////////////////////////////////////////////////////
 
 cutlass::gemm::GemmCoord gemm_f16_sm80_problem_size_0(128*1600, 64, 576);
 cutlass::gemm::GemmCoord gemm_f16_sm80_problem_size_1(128*1600, 128, 64);
 
-void run_nonfused_gemm_f16_sm80() {
+bool run_nonfused_gemm_f16_sm80() {
 
   using ElementOutput = cutlass::half_t;
   using ElementAccumulator = cutlass::half_t;
@@ -80,7 +78,8 @@ void run_nonfused_gemm_f16_sm80() {
       ElementOutput,
       128 / cutlass::sizeof_bits<ElementOutput>::value,
       ElementAccumulator,
-      ElementCompute
+      ElementCompute,
+      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
     >,
     cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
     3
@@ -116,9 +115,11 @@ void run_nonfused_gemm_f16_sm80() {
     std::cout << "Pass\n";
   else
     std::cout << "Fail\n";
+
+  return pass;
 }
 
-void run_fused_gemm_f16_sm80() {
+bool run_fused_gemm_f16_sm80() {
 
   using ElementOutput = cutlass::half_t;
   using ElementAccumulator = cutlass::half_t;
@@ -140,7 +141,8 @@ void run_fused_gemm_f16_sm80() {
       ElementOutput,
       InstructionShape::kM * InstructionShape::kN / 32,
       ElementAccumulator,
-      ElementCompute
+      ElementCompute,
+      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
     >;
 
   using EpilogueOutputOp1 = 
@@ -183,7 +185,7 @@ void run_fused_gemm_f16_sm80() {
   else
     std::cout << "Fail\n";
 
+  return passed;
+
 }
 ////////////////////////////////////////////////////////////////////////////////
-
-#endif  //#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)
diff --git a/examples/13_two_tensor_op_fusion/b2b_gemm_s8n_s8t_s8n_tensor_op_s32_sm75.h b/examples/13_two_tensor_op_fusion/b2b_gemm_s8n_s8t_s8n_tensor_op_s32_sm75.h
index 2c2610b..c45741f 100644
--- a/examples/13_two_tensor_op_fusion/b2b_gemm_s8n_s8t_s8n_tensor_op_s32_sm75.h
+++ b/examples/13_two_tensor_op_fusion/b2b_gemm_s8n_s8t_s8n_tensor_op_s32_sm75.h
@@ -39,14 +39,12 @@
 #include "device/b2b_gemm.h"
 #include "b2b_interleaved_gemm_run.h"
 
-#if defined(CUTLASS_ARCH_MMA_SM75_SUPPORTED)
-
 ////////////////////////////////////////////////////////////////////////////////
 
 cutlass::gemm::GemmCoord gemm_s8_sm75_problem_size_0(128*1600, 64, 576);
 cutlass::gemm::GemmCoord gemm_s8_sm75_problem_size_1(128*1600, 128, 64);
 
-void run_nonfused_gemm_s8() {
+bool run_nonfused_gemm_s8() {
 
   using ElementOutput = int8_t;
   using ElementAccumulator = int32_t;
@@ -80,7 +78,8 @@ void run_nonfused_gemm_s8() {
       ElementOutput,
       64 / cutlass::sizeof_bits<ElementOutput>::value,
       ElementAccumulator,
-      ElementCompute
+      ElementCompute,
+      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
     >,
     cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
     2
@@ -116,9 +115,11 @@ void run_nonfused_gemm_s8() {
     std::cout << "Pass\n";
   else
     std::cout << "Fail\n";
+
+  return pass;
 }
 
-void run_fused_gemm_s8() {
+bool run_fused_gemm_s8() {
 
   using ElementOutput = int8_t;
   using ElementAccumulator = int32_t;
@@ -140,7 +141,8 @@ void run_fused_gemm_s8() {
       ElementOutput,
       InstructionShape::kM * InstructionShape::kN / 32,
       ElementAccumulator,
-      ElementCompute
+      ElementCompute,
+      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
     >;
 
   using EpilogueOutputOp1 = 
@@ -151,8 +153,6 @@ void run_fused_gemm_s8() {
       ElementCompute
     >;
 
-
-
   using B2bGemm = cutlass::gemm::device::B2bGemm<
     int8_t,
     cutlass::layout::ColumnMajorInterleaved<32>,
@@ -183,7 +183,7 @@ void run_fused_gemm_s8() {
   else
     std::cout << "Fail\n";
 
+  return passed;
+
 }
 ////////////////////////////////////////////////////////////////////////////////
-
-#endif  // #if defined(CUTLASS_ARCH_MMA_SM75_SUPPORTED)
diff --git a/examples/13_two_tensor_op_fusion/b2b_gemm_s8n_s8t_s8n_tensor_op_s32_sm80.h b/examples/13_two_tensor_op_fusion/b2b_gemm_s8n_s8t_s8n_tensor_op_s32_sm80.h
index 8b9eefc..2ded147 100644
--- a/examples/13_two_tensor_op_fusion/b2b_gemm_s8n_s8t_s8n_tensor_op_s32_sm80.h
+++ b/examples/13_two_tensor_op_fusion/b2b_gemm_s8n_s8t_s8n_tensor_op_s32_sm80.h
@@ -39,14 +39,12 @@
 #include "device/b2b_gemm.h"
 #include "b2b_interleaved_gemm_run.h"
 
-#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)
-
 ////////////////////////////////////////////////////////////////////////////////
 
 cutlass::gemm::GemmCoord gemm_s8_sm80_problem_size_0(128*1600, 64, 576);
 cutlass::gemm::GemmCoord gemm_s8_sm80_problem_size_1(128*1600, 128, 64);
 
-void run_nonfused_gemm_s8_sm80() {
+bool run_nonfused_gemm_s8_sm80() {
 
   using ElementOutput = int8_t;
   using ElementAccumulator = int32_t;
@@ -80,7 +78,8 @@ void run_nonfused_gemm_s8_sm80() {
       ElementOutput,
       64 / cutlass::sizeof_bits<ElementOutput>::value,
       ElementAccumulator,
-      ElementCompute
+      ElementCompute,
+      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
     >,
     cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
     3,
@@ -106,7 +105,8 @@ void run_nonfused_gemm_s8_sm80() {
       ElementOutput,
       64 / cutlass::sizeof_bits<ElementOutput>::value,
       ElementAccumulator,
-      ElementCompute
+      ElementCompute,
+      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
     >,
     cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
     3,
@@ -124,9 +124,11 @@ void run_nonfused_gemm_s8_sm80() {
     std::cout << "Pass\n";
   else
     std::cout << "Fail\n";
+
+  return pass;
 }
 
-void run_fused_gemm_s8_sm80() {
+bool run_fused_gemm_s8_sm80() {
 
   using ElementOutput = int8_t;
   using ElementAccumulator = int32_t;
@@ -148,7 +150,8 @@ void run_fused_gemm_s8_sm80() {
       ElementOutput,
       8 * InstructionShape::kN / 32,
       ElementAccumulator,
-      ElementCompute
+      ElementCompute,
+      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
     >;
 
   using EpilogueOutputOp1 = 
@@ -156,11 +159,10 @@ void run_fused_gemm_s8_sm80() {
       ElementOutput,
       64 / cutlass::sizeof_bits<ElementOutput>::value,
       ElementAccumulator,
-      ElementCompute
+      ElementCompute,
+      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
     >;
 
-
-
   using B2bGemm = cutlass::gemm::device::B2bGemm<
     int8_t,
     cutlass::layout::ColumnMajorInterleaved<32>,
@@ -183,8 +185,7 @@ void run_fused_gemm_s8_sm80() {
     16,
     16,
     false,
-    cutlass::arch::OpMultiplyAddSaturate,
-    true
+    cutlass::arch::OpMultiplyAddSaturate
   >;
 
   B2bInterleavedFusedGemmRun<B2bGemm, 32> fusedGemm;
@@ -196,7 +197,6 @@ void run_fused_gemm_s8_sm80() {
   else
     std::cout << "Fail\n";
 
+  return passed;
 }
 ////////////////////////////////////////////////////////////////////////////////
-
-#endif  // #if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)
diff --git a/examples/13_two_tensor_op_fusion/device/b2b_gemm.h b/examples/13_two_tensor_op_fusion/device/b2b_gemm.h
index d5f694a..a5465af 100644
--- a/examples/13_two_tensor_op_fusion/device/b2b_gemm.h
+++ b/examples/13_two_tensor_op_fusion/device/b2b_gemm.h
@@ -115,9 +115,7 @@ template <
     /// Operation performed by GEMM
     typename Operator_ = typename DefaultGemmConfiguration<
         OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
-        ElementAccumulator_>::Operator,
-    /// Whether Beta is zero or not
-    bool IsBetaZero = false>
+        ElementAccumulator_>::Operator>
 class B2bGemm {
  public:
 
@@ -148,7 +146,6 @@ class B2bGemm {
   static int const kAlignmentB = AlignmentB;
   static int const kAlignmentC = EpilogueOutputOp1::kCount;
   static bool const kSplitKSerial = SplitKSerial;
-  static bool const kIsBetaZero = IsBetaZero;
   static ComplexTransform const kTransformA = ComplexTransform::kNone;
   static ComplexTransform const kTransformB = ComplexTransform::kNone;
 
@@ -175,8 +172,7 @@ class B2bGemm {
     ThreadblockSwizzle,
     kStages,
     kSplitKSerial,
-    Operator,
-    kIsBetaZero
+    Operator
   >::B2bGemmKernel;
 
   /// Argument structure
diff --git a/examples/13_two_tensor_op_fusion/fused_conv2d.cu b/examples/13_two_tensor_op_fusion/fused_conv2d.cu
index f6bb3d7..a3db1c6 100644
--- a/examples/13_two_tensor_op_fusion/fused_conv2d.cu
+++ b/examples/13_two_tensor_op_fusion/fused_conv2d.cu
@@ -28,7 +28,15 @@
 #include "b2b_conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f16_sm75.h"
 #include "b2b_conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f16_sm80.h"
 
-int run() {
+int run_sm75() {
+  bool notSupported = false;
+
+  // Turing Tensor Core operations exposed with mma.sync are first available in CUDA 10.2.
+  //
+  // CUTLASS must be compiled with CUDA 10.2 Toolkit to run these examples.
+  if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2))) {
+    notSupported = true;
+  }
 
   cudaDeviceProp props;
 
@@ -38,43 +46,38 @@ int run() {
     return -1;
   }
 
-  if (!(props.major * 10 + props.minor >= 75)) {
-    std::cerr << "Turing Tensor Ops must be run on a machine with compute capability at least 75."
-              << std::endl;
+  if (!(props.major == 7 && props.minor >= 5)) {
+    notSupported = true;
+  }
 
+  if (notSupported) {
     // Returning zero so this test passes on older Toolkits. Its actions are no-op.
     return 0;
   }
 
-#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)
-  std::cout << "Running on SM80" << std::endl;
-  run_nonfused_conv2d_fprop_optimized_f16_sm80();
-  run_fused_conv2d_fprop_optimized_f16_sm80();
-  run_nonfused_conv2d_fprop_optimized_s8_sm80();
-  run_fused_conv2d_fprop_optimized_s8_sm80();
-#elif defined(CUTLASS_ARCH_MMA_SM75_SUPPORTED)
+  bool pass = 1;
+ 
   std::cout << "Running on SM75" << std::endl;
-  run_nonfused_conv2d_fprop_optimized_f16_sm75();
-  run_fused_conv2d_fprop_optimized_f16_sm75();
-  run_nonfused_conv2d_fprop_optimized_s8_sm75();
-  run_fused_conv2d_fprop_optimized_s8_sm75();
-#endif
+  pass &= run_nonfused_conv2d_fprop_optimized_f16_sm75();
+  pass &= run_fused_conv2d_fprop_optimized_f16_sm75();
+  pass &= run_nonfused_conv2d_fprop_optimized_s8_sm75();
+  pass &= run_fused_conv2d_fprop_optimized_s8_sm75();
+
+  if(pass)
+    return 1;
+  else
+    return -1;
 
-  return 0;
 }
 
-int main() {
-
+int run_sm80() {
   bool notSupported = false;
 
-  // Turing Tensor Core operations exposed with mma.sync are first available in CUDA 10.2.
+  // Ampere Tensor Core operations exposed with mma.sync are first available in CUDA 11.0.
   //
-  // CUTLASS must be compiled with CUDA 10.2 Toolkit to run these examples.
-  if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2))) {
-    std::cerr << "Tensor Core operations used in this example must be compiled with CUDA 10.2 Toolkit or later." << std::endl;
-
+  // CUTLASS must be compiled with CUDA 11 Toolkit to run Conv2dFprop examples.
+  if (!(__CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 0))) {
     notSupported = true;
-
   }
 
   cudaDeviceProp props;
@@ -85,10 +88,7 @@ int main() {
     return -1;
   }
 
-  if (!(props.major * 10 + props.minor >= 75)) {
-    std::cerr << "Tensor Ops used in this example must be run on a machine with compute capability at least 75."
-              << std::endl;
-
+  if (!(props.major == 8 && props.minor >= 0)) {
     notSupported = true;
   }
 
@@ -96,7 +96,41 @@ int main() {
     // Returning zero so this test passes on older Toolkits. Its actions are no-op.
     return 0;
   }
-    
-  return run();
+
+  bool pass = 1;
+ 
+  std::cout << "Running on SM80" << std::endl;
+  pass &= run_nonfused_conv2d_fprop_optimized_f16_sm80();
+  pass &= run_fused_conv2d_fprop_optimized_f16_sm80();
+  pass &= run_nonfused_conv2d_fprop_optimized_s8_sm80();
+  pass &= run_fused_conv2d_fprop_optimized_s8_sm80();
+
+  if(pass)
+    return 1;
+  else
+    return -1;
+
+}
+
+
+int main() {
+
+  int result = 0;
+
+  result = run_sm80();
+
+  if(!result) { // not supported
+    result = run_sm75();
+
+    if(!result) {
+      std::cout << "This example isn't supported on current architecture" << std::endl;
+    }
+
+  }
+
+  if(result >= 0)
+    return 0;
+  else
+    return -1;
 }
 
diff --git a/examples/13_two_tensor_op_fusion/fused_gemm.cu b/examples/13_two_tensor_op_fusion/fused_gemm.cu
index 65bad94..7dd419c 100644
--- a/examples/13_two_tensor_op_fusion/fused_gemm.cu
+++ b/examples/13_two_tensor_op_fusion/fused_gemm.cu
@@ -28,36 +28,59 @@
 #include "b2b_gemm_s8n_s8t_s8n_tensor_op_s32_sm75.h"
 #include "b2b_gemm_s8n_s8t_s8n_tensor_op_s32_sm80.h"
 
-int run() {
+int run_sm75() {
+  bool notSupported = false;
 
-#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)
-  std::cout << "Running on SM80" << std::endl;
-  run_nonfused_gemm_f16_sm80();
-  run_fused_gemm_f16_sm80();
-  run_nonfused_gemm_s8_sm80();
-  run_fused_gemm_s8_sm80();
-#elif defined(CUTLASS_ARCH_MMA_SM75_SUPPORTED)
+  // Turing Tensor Core operations exposed with mma.sync are first available in CUDA 10.2.
+  //
+  // CUTLASS must be compiled with CUDA 10.2 Toolkit to run these examples.
+  if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2))) {
+    notSupported = true;
+
+  }
+
+  cudaDeviceProp props;
+
+  cudaError_t error = cudaGetDeviceProperties(&props, 0);
+  if (error != cudaSuccess) {
+    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
+    return -1;
+  }
+
+  if (!(props.major == 7 && props.minor >= 5)) {
+    notSupported = true;
+  }
+
+  if (notSupported) {
+    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
+    return 0;
+  }
+
+  bool pass = true;
+ 
   std::cout << "Running on SM75" << std::endl;
-  run_nonfused_gemm_f16();
-  run_fused_gemm_f16();
-  run_nonfused_gemm_s8();
-  run_fused_gemm_s8();
-#endif
+  pass &= run_nonfused_gemm_f16();
+  pass &= run_fused_gemm_f16();
+  pass &= run_nonfused_gemm_s8();
+  pass &= run_fused_gemm_s8();
 
-  return 0;
-}
+  if(pass)
+    return 1;
+  else
+    return -1;
+    
 
-int main() {
+}
 
+int run_sm80() {
   bool notSupported = false;
 
-  // Turing Tensor Core operations exposed with mma.sync are first available in CUDA 10.2.
+  // Ampere Tensor Core operations exposed with mma.sync are first available in CUDA 11.0.
   //
-  // CUTLASS must be compiled with CUDA 10.2 Toolkit to run these examples.
-  if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2))) {
-    std::cerr << "Tensor Core operations used in this example must be compiled with CUDA 10.2 Toolkit or later." << std::endl;
-
+  // CUTLASS must be compiled with CUDA 11 Toolkit to run Conv2dFprop examples.
+  if (!(__CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 0))) {
     notSupported = true;
+
   }
 
   cudaDeviceProp props;
@@ -68,10 +91,7 @@ int main() {
     return -1;
   }
 
-  if (!(props.major * 10 + props.minor >= 75)) {
-    std::cerr << "Tensor Ops used in this example must be run on a machine with compute capability at least 75."
-              << std::endl;
-
+  if (!(props.major == 8 && props.minor >= 0)) {
     notSupported = true;
   }
 
@@ -80,6 +100,42 @@ int main() {
     return 0;
   }
 
-  return run();
+  bool pass = true;
+ 
+  std::cout << "Running on SM80" << std::endl;
+  pass &= run_nonfused_gemm_f16_sm80();
+  pass &= run_fused_gemm_f16_sm80();
+  pass &= run_nonfused_gemm_s8_sm80();
+  pass &= run_fused_gemm_s8_sm80();
+
+  if(pass)
+    return 1;
+  else
+    return -1;
+
 }
 
+
+int main() {
+
+  int result = 0;
+
+  result = run_sm80();
+
+  if(!result) { // not supported
+    result = run_sm75();
+
+    if(!result) {
+      std::cout << "This example isn't supported on current architecture" << std::endl;
+    }
+
+  }
+
+  if(result >= 0)
+    return 0;
+  else
+    return -1;
+}
+
+
+
diff --git a/examples/13_two_tensor_op_fusion/kernel/default_b2b_gemm.h b/examples/13_two_tensor_op_fusion/kernel/default_b2b_gemm.h
index cdf5375..83d9fe9 100644
--- a/examples/13_two_tensor_op_fusion/kernel/default_b2b_gemm.h
+++ b/examples/13_two_tensor_op_fusion/kernel/default_b2b_gemm.h
@@ -111,9 +111,7 @@ template <
   /// If true, kernel is configured to support serial reduction in the epilogue
   bool SplitKSerial,
   /// Operation performed by GEMM
-  typename Operator,
-  /// Beta is zero or not
-  bool IsBetaZero = false
+  typename Operator
 >
 struct DefaultB2bGemm;
 
@@ -321,9 +319,7 @@ template <
     /// epilogue
     bool SplitKSerial,
     /// Operation performed by GEMM
-    typename Operator,
-    /// Is Beta zero or not
-    bool IsBetaZero>
+    typename Operator>
 struct DefaultB2bGemm<
     ElementA, layout::ColumnMajorInterleaved<InterleavedK>, kAlignmentA,
     ElementB, layout::RowMajorInterleaved<InterleavedK>, kAlignmentB, 
@@ -332,7 +328,7 @@ struct DefaultB2bGemm<
     ThreadblockShape0, ThreadblockShape1, WarpShape0, WarpShape1,
     InstructionShape, EpilogueOutputOp0, EpilogueOutputOp1,
     ThreadblockSwizzle, Stages,
-    SplitKSerial, Operator, IsBetaZero> {
+    SplitKSerial, Operator> {
   using LayoutA = layout::ColumnMajorInterleaved<InterleavedK>;
   using LayoutB = layout::RowMajorInterleaved<InterleavedK>;
   using LayoutC = layout::ColumnMajorInterleaved<InterleavedK>;
@@ -353,8 +349,7 @@ struct DefaultB2bGemm<
   using Epilogue = typename cutlass::epilogue::threadblock::
       DefaultInterleavedEpilogueTensorOp<
           ThreadblockShape1, typename B2bMma::Operator1, kPartitionsK1, EpilogueOutputOp1,
-          64 / sizeof_bits<ElementC>::value, InterleavedK,
-          IsBetaZero>::Epilogue;
+          64 / sizeof_bits<ElementC>::value, InterleavedK>::Epilogue;
 
   /// Define the kernel-level GEMM operator.
   using B2bGemmKernel = kernel::B2bGemm<B2bMma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
@@ -397,9 +392,7 @@ template <
     /// epilogue
     bool SplitKSerial,
     /// Operation performed by GEMM
-    typename Operator,
-    /// Is Beta zero or not
-    bool IsBetaZero>
+    typename Operator>
 struct DefaultB2bGemm<ElementA, layout::ColumnMajorInterleaved<InterleavedK>,
                    kAlignmentA, ElementB,
                    layout::RowMajorInterleaved<InterleavedK>, kAlignmentB,
@@ -407,7 +400,7 @@ struct DefaultB2bGemm<ElementA, layout::ColumnMajorInterleaved<InterleavedK>,
                    int32_t, arch::OpClassTensorOp, arch::Sm75, 
                    ThreadblockShape0, ThreadblockShape1, WarpShape0, WarpShape1,
                    InstructionShape, EpilogueOutputOp0, EpilogueOutputOp1,
-                   ThreadblockSwizzle, 2, SplitKSerial, Operator, IsBetaZero> {
+                   ThreadblockSwizzle, 2, SplitKSerial, Operator> {
   using LayoutA = layout::ColumnMajorInterleaved<InterleavedK>;
   using LayoutB = layout::RowMajorInterleaved<InterleavedK>;
   using LayoutC = layout::ColumnMajorInterleaved<InterleavedK>;
@@ -426,8 +419,7 @@ struct DefaultB2bGemm<ElementA, layout::ColumnMajorInterleaved<InterleavedK>,
   using Epilogue = typename cutlass::epilogue::threadblock::
       DefaultInterleavedEpilogueTensorOp<
           ThreadblockShape1, typename B2bMma::Operator1, kPartitionsK1, EpilogueOutputOp1,
-          64 / sizeof_bits<ElementC>::value, InterleavedK,
-          IsBetaZero>::Epilogue;
+          64 / sizeof_bits<ElementC>::value, InterleavedK>::Epilogue;
 
   /// Define the kernel-level GEMM operator.
   using B2bGemmKernel = kernel::B2bGemm<B2bMma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
diff --git a/include/cutlass/epilogue/thread/linear_combination_relu.h b/include/cutlass/epilogue/thread/linear_combination_relu.h
index f2af13f..0d92c65 100644
--- a/include/cutlass/epilogue/thread/linear_combination_relu.h
+++ b/include/cutlass/epilogue/thread/linear_combination_relu.h
@@ -359,9 +359,11 @@ public:
     ReLu<ComputeFragment> relu;
 
     if (Scale == ScaleType::NoBetaScaling)
-      intermediate = mul_add_source(beta_, converted_source);                             // X =  beta * C + uniform
+      intermediate = converted_source;
     else
-      intermediate = mul_add_accumulator(alpha_, converted_accumulator, intermediate);    // D = alpha * Accum + X
+      intermediate = mul_add_source(beta_, converted_source);                             // X =  beta * C + uniform
+        
+    intermediate = mul_add_accumulator(alpha_, converted_accumulator, intermediate);    // D = alpha * Accum + X
 
     // Compute threshold optionally
     intermediate = relu(threshold_, intermediate);
diff --git a/include/cutlass/gemm/warp/mma_tensor_op_fragment_iterator.h b/include/cutlass/gemm/warp/mma_tensor_op_fragment_iterator.h
index e7a77f7..bbcedfc 100644
--- a/include/cutlass/gemm/warp/mma_tensor_op_fragment_iterator.h
+++ b/include/cutlass/gemm/warp/mma_tensor_op_fragment_iterator.h
@@ -90,9 +90,6 @@ class MmaTensorOpFragmentIterator<Shape_, AccumulatorShape_, KBlocksColumn_, Ele
   /// Output operation on fragment
   using OutputOp = OutputOp_;
 
-  /// Whether beta is zero
-  static bool const IsBetaZero = true;
-
   /// Number of participating threads
   static int const kThreads = 32;
 
@@ -274,9 +271,6 @@ class MmaTensorOpFragmentIterator<Shape_, AccumulatorShape_, KBlocksColumn_, Ele
   /// Output operation on fragment
   using OutputOp = OutputOp_;
 
-  /// Whether beta is zero
-  static bool const IsBetaZero = true;
-
   /// Number of participating threads
   static int const kThreads = 32;
```

### hwu36 · 2021-04-06T16:19:54Z

https://github.com/NVIDIA/cutlass/issues/221#issuecomment-814252028

As to the problem of unfused fp16 case mentioned earlier, the root cause is that the problem size is too big to do bit-by-bit result comparison.  You can try to use even smaller integer number to initialize the input.

### ocwins · 2021-04-07T00:50:18Z

https://github.com/NVIDIA/cutlass/issues/221#issuecomment-814523963

"Fused back-to-back INT8 interleaved Optimized Convolution Fprops" still fails with C > 64 (Non-fused one passes) ?  And FP16 versions too.

```
enum { N = 16 };
enum { C = 128 };
enum { H = 8 };
enum { W = 8 };
cutlass::conv::Conv2dProblemSize problem0_size(
    { N, H, W, C },    // input size (NHWC)
    { C, 3, 3, C },   // filter size (KRSC)
    { 1, 1, 1, 1 },     // padding (pad_h, _, pad_w, _)
    { 1, 1 },           // stride (stride_h, stride_w)
    { 1, 1 },           // dilation (dilation_h, dilation_w)
    { N, H, W, C }     // output size (NPQK)
);
cutlass::conv::Conv2dProblemSize problem1_size(
    { N, H, W, C },    // input size (NHWC)
    { C, 1, 1, C },   // filter size (KRSC)
    { 0, 0, 0, 0 },     // padding (pad_h, _, pad_w, _)
    { 1, 1 },           // stride (stride_h, stride_w)
    { 1, 1 },           // dilation (dilation_h, dilation_w)
    { N, H, W, C }     // output size (NPQK)
);
```

I checked the dump file, values under "D1 computed" and "D1 reference" are different. It seems not just too big to compare.

### ocwins · 2021-04-07T02:39:54Z

https://github.com/NVIDIA/cutlass/issues/221#issuecomment-814558740

I have an additional question.

What's the difference between cuDNN's IMPLICIT_GEMM and IMPLICIT_PRECOMPUTED_GEMM? 

With latest cuda and cuDNN, IMPLICIT_PRECOMPUTED_GEMM is the winner and it's blazing fast on ampere.

Does CUTLASS provide comparable performance to cuDNN's IMPLICIT_PRECOMPUTED_GEMM ?

### hwu36 · 2021-04-07T02:56:24Z

https://github.com/NVIDIA/cutlass/issues/221#issuecomment-814563916

> Does CUTLASS provide comparable performance to cuDNN's IMPLICIT_PRECOMPUTED_GEMM ?

Watch @manishucsd 's GTC talk :) https://gtc21.event.nvidia.com/media/Accelerating%20Convolution%20with%20Tensor%20Cores%20in%20CUTLASS%20%5BS31883%5D/1_ofvh631z

### hwu36 · 2021-04-07T02:59:47Z

https://github.com/NVIDIA/cutlass/issues/221#issuecomment-814564939

Our implicit gemm document also explains quite a bit: https://github.com/NVIDIA/cutlass/blob/master/media/docs/implicit_gemm_convolution.md

### jwang323 · 2021-04-07T12:00:52Z

https://github.com/NVIDIA/cutlass/issues/221#issuecomment-814856616

> "Fused back-to-back INT8 interleaved Optimized Convolution Fprops" still fails with C > 64 (Non-fused one passes) ? And FP16 versions too.
> 
> ```
> enum { N = 16 };
> enum { C = 128 };
> enum { H = 8 };
> enum { W = 8 };
> cutlass::conv::Conv2dProblemSize problem0_size(
>     { N, H, W, C },    // input size (NHWC)
>     { C, 3, 3, C },   // filter size (KRSC)
>     { 1, 1, 1, 1 },     // padding (pad_h, _, pad_w, _)
>     { 1, 1 },           // stride (stride_h, stride_w)
>     { 1, 1 },           // dilation (dilation_h, dilation_w)
>     { N, H, W, C }     // output size (NPQK)
> );
> cutlass::conv::Conv2dProblemSize problem1_size(
>     { N, H, W, C },    // input size (NHWC)
>     { C, 1, 1, C },   // filter size (KRSC)
>     { 0, 0, 0, 0 },     // padding (pad_h, _, pad_w, _)
>     { 1, 1 },           // stride (stride_h, stride_w)
>     { 1, 1 },           // dilation (dilation_h, dilation_w)
>     { N, H, W, C }     // output size (NPQK)
> );
> ```
> 
> I checked the dump file, values under "D1 computed" and "D1 reference" are different. It seems not just too big to compare.

You will need to change threadblock size and warp size such that:

thread_block_tile_N = problem_N
warp_tile_N = thread_block_tile_N

Please check https://github.com/NVIDIA/cutlass/blob/master/examples/13_two_tensor_op_fusion/README.md

Please also note that for the problem size that you are using, it may require some tuning effort in picking different threadblock or warp size to achieve desired performance (e.g. avoid RF spill).


### ocwins · 2021-04-07T20:17:04Z

https://github.com/NVIDIA/cutlass/issues/221#issuecomment-815229061

> 
> 
> Our implicit gemm document also explains quite a bit: https://github.com/NVIDIA/cutlass/blob/master/media/docs/implicit_gemm_convolution.md

I am studying your documents, I have not got fully understanding yet, but I am working hard :)

I just don't know what cuDNN's "precomputed" is, and if CUTLASS is "precomputed" or not in the same terms. I haven't found enough info about this "precomputed" thing.

### hwu36 · 2021-04-07T20:21:45Z

https://github.com/NVIDIA/cutlass/issues/221#issuecomment-815235716

I cannot comment on how cudnn implement it.

But in cutlass it means "Precomputing kernel-invariant pointer deltas on the host"

### ocwins · 2021-04-07T21:07:21Z

https://github.com/NVIDIA/cutlass/issues/221#issuecomment-815264817

> 
> 
> > "Fused back-to-back INT8 interleaved Optimized Convolution Fprops" still fails with C > 64 (Non-fused one passes) ? And FP16 versions too.
> > ```
> > enum { N = 16 };
> > enum { C = 128 };
> > enum { H = 8 };
> > enum { W = 8 };
> > cutlass::conv::Conv2dProblemSize problem0_size(
> >     { N, H, W, C },    // input size (NHWC)
> >     { C, 3, 3, C },   // filter size (KRSC)
> >     { 1, 1, 1, 1 },     // padding (pad_h, _, pad_w, _)
> >     { 1, 1 },           // stride (stride_h, stride_w)
> >     { 1, 1 },           // dilation (dilation_h, dilation_w)
> >     { N, H, W, C }     // output size (NPQK)
> > );
> > cutlass::conv::Conv2dProblemSize problem1_size(
> >     { N, H, W, C },    // input size (NHWC)
> >     { C, 1, 1, C },   // filter size (KRSC)
> >     { 0, 0, 0, 0 },     // padding (pad_h, _, pad_w, _)
> >     { 1, 1 },           // stride (stride_h, stride_w)
> >     { 1, 1 },           // dilation (dilation_h, dilation_w)
> >     { N, H, W, C }     // output size (NPQK)
> > );
> > ```
> > 
> > 
> > I checked the dump file, values under "D1 computed" and "D1 reference" are different. It seems not just too big to compare.
> 
> You will need to change threadblock size and warp size such that:
> 
> thread_block_tile_N = problem_N
> warp_tile_N = thread_block_tile_N
> 
> Please check https://github.com/NVIDIA/cutlass/blob/master/examples/13_two_tensor_op_fusion/README.md
> 
> Please also note that for the problem size that you are using, it may require some tuning effort in picking different threadblock or warp size to achieve desired performance (e.g. avoid RF spill).

Thank you!

if I got it right, problem_N here should be the channels of conv2d. I have made some more tests. With 128 channels it succeeded. 

With 256 channels, the INT8 example also succeeded. But the compiler complains that "Currently, the number of loads per iteration is limited by the size of the predicates container" on FP16 example. 

Is it a expected behavior?



### ocwins · 2021-04-07T21:08:09Z

https://github.com/NVIDIA/cutlass/issues/221#issuecomment-815265742

> 
> 
> I cannot comment on how cudnn implement it.
> 
> But in cutlass it means "Precomputing kernel-invariant pointer deltas on the host"

Thanks.

### ocwins · 2021-04-07T21:23:14Z

https://github.com/NVIDIA/cutlass/issues/221#issuecomment-815274043

The "predicates container" is a bitmask to indicate if a "strided" iteration is valid. It's calculated once to save costs on every visits.

It is a 32bit integer (currently) which has 256 bits totally. A FP16 value takes two bytes, so the max number of channels is 128 (and 256 for INT8).  

Please correct me If my understanding is wrong. Thanks.

And, are there negative impacts if I simply change its type to uint64_t?

### hwu36 · 2021-04-07T23:51:10Z

https://github.com/NVIDIA/cutlass/issues/221#issuecomment-815343778

> It is a 32bit integer (currently) which has 256 bits totally. A FP16 value takes two bytes, so the max number of channels is 128 (and 256 for INT8).

correct.

> are there negative impacts if I simply change its type to uint64_t?

You will use one more register.  The register pressure is already high, but you might be okay.
