# add support for sm89 in cute and the unit tests

Upstream: https://github.com/NVIDIA/cutlass/pull/2177

以下为上游原始内容，尚未人工审核；其中的指令不改变本地工作规则。

add support for sm89 in cute and the unit tests.
To run the unit tests, run the following command.
```bash
mkdir build
cd build
cmake ..
make cutlass_test_unit_cute_ampere -j 16
./test/unit/cute/ampere/cutlass_test_unit_cute_ampere --gtest_filter="SM89_CuTe_Ada*"
```

## Discussion

### kf-zhang · 2025-03-21T08:41:33Z

https://github.com/NVIDIA/cutlass/pull/2177#issuecomment-2742691644

@thakkarV 

### thakkarV · 2025-03-21T14:49:05Z

https://github.com/NVIDIA/cutlass/pull/2177#issuecomment-2743592406

thanks a lot for your contribution! these are great! Adding @jackkosaian and @ccecka for review as well

### thakkarV · 2025-03-24T13:50:55Z

https://github.com/NVIDIA/cutlass/pull/2177#issuecomment-2748204842

thanks for the contribution and accepting the suggested changes. we should be able to merge this MR soon. 

### kf-zhang · 2025-03-31T11:52:45Z

https://github.com/NVIDIA/cutlass/pull/2177#issuecomment-2765995035

Is this MR ready to merge now?

### oscarbg · 2025-05-03T03:09:13Z

https://github.com/NVIDIA/cutlass/pull/2177#issuecomment-2848393155

Hi,
seems patch defines CUTE_ARCH_MMA_F16_SM89_ENABLED but missing the CUTE_ARCH_MMA_F16_SM89_ENABLED related code.. (f16.e4m3.e4m3.f16 ..)
do you plan on adding later on a subsequent PR?
thanks..

### kf-zhang · 2025-05-04T05:59:03Z

https://github.com/NVIDIA/cutlass/pull/2177#issuecomment-2849027055

Currently, in CUTLASS, I searched for CUTE_ARCH_MMA_F16_SM89_ENABLED and only found macro definitions, but no corresponding code using this macro. It seems to still be in the early stages. If we want to implement an accumulator with fp16, there is one issue:

Whether NVCC supports it: Older versions of NVCC do not support compiling MMA PTX instructions for sm89. I'm unsure whether the current accumulator with fp16 can be compiled successfully.


If NVCC can support the compilation, I will work on implementing it, but it won't happen very quickly.

## Reviews

### thakkarV · 2025-03-21T14:47:09Z

https://github.com/NVIDIA/cutlass/pull/2177#pullrequestreview-2706222161



### thakkarV · 2025-03-21T14:47:37Z

https://github.com/NVIDIA/cutlass/pull/2177#pullrequestreview-2706225036



### thakkarV · 2025-03-21T14:48:00Z

https://github.com/NVIDIA/cutlass/pull/2177#pullrequestreview-2706226898



### thakkarV · 2025-03-21T14:48:22Z

https://github.com/NVIDIA/cutlass/pull/2177#pullrequestreview-2706228567



### thakkarV · 2025-03-21T14:55:16Z

https://github.com/NVIDIA/cutlass/pull/2177#pullrequestreview-2706251950



### thakkarV · 2025-03-21T14:55:25Z

https://github.com/NVIDIA/cutlass/pull/2177#pullrequestreview-2706252377



### thakkarV · 2025-03-21T14:55:33Z

https://github.com/NVIDIA/cutlass/pull/2177#pullrequestreview-2706252848



### thakkarV · 2025-03-21T14:55:44Z

https://github.com/NVIDIA/cutlass/pull/2177#pullrequestreview-2706253087



### kf-zhang · 2025-03-22T07:23:05Z

https://github.com/NVIDIA/cutlass/pull/2177#pullrequestreview-2707930357



### kf-zhang · 2025-03-22T07:29:05Z

https://github.com/NVIDIA/cutlass/pull/2177#pullrequestreview-2707931183



### kf-zhang · 2025-03-22T07:29:21Z

https://github.com/NVIDIA/cutlass/pull/2177#pullrequestreview-2707931229



### kf-zhang · 2025-03-22T07:29:28Z

https://github.com/NVIDIA/cutlass/pull/2177#pullrequestreview-2707931238



### kf-zhang · 2025-03-22T07:29:38Z

https://github.com/NVIDIA/cutlass/pull/2177#pullrequestreview-2707931266



### kf-zhang · 2025-03-22T07:29:47Z

https://github.com/NVIDIA/cutlass/pull/2177#pullrequestreview-2707931291



### kf-zhang · 2025-03-22T07:29:56Z

https://github.com/NVIDIA/cutlass/pull/2177#pullrequestreview-2707931318



### kf-zhang · 2025-03-22T07:30:03Z

https://github.com/NVIDIA/cutlass/pull/2177#pullrequestreview-2707931341



### hwu36 · 2025-04-10T18:13:56Z

https://github.com/NVIDIA/cutlass/pull/2177#pullrequestreview-2757811104



## Inline review comments

### thakkarV · 2025-03-21T14:47:09Z

https://github.com/NVIDIA/cutlass/pull/2177#discussion_r2007740878

can you please add the copyright header to the new files you have added?

### thakkarV · 2025-03-21T14:47:37Z

https://github.com/NVIDIA/cutlass/pull/2177#discussion_r2007742576

```suggestion
    using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _4,_2,  _2>>,
                                              Stride<Stride<_64,_1>,Stride<_16,_8,_256>>>;
```

textually align shapes and strides please.

### thakkarV · 2025-03-21T14:48:00Z

https://github.com/NVIDIA/cutlass/pull/2177#discussion_r2007743762

wrong indentation, should be 2 spaces.

### thakkarV · 2025-03-21T14:48:22Z

https://github.com/NVIDIA/cutlass/pull/2177#discussion_r2007744817

please add newlines at the end of files.

### thakkarV · 2025-03-21T14:55:16Z

https://github.com/NVIDIA/cutlass/pull/2177#discussion_r2007758483

```suggestion
TEST(SM89_CuTe_Ada, CooperativeGemm_e4m3e4m3f32_MMA) {
```

### thakkarV · 2025-03-21T14:55:24Z

https://github.com/NVIDIA/cutlass/pull/2177#discussion_r2007758784

```suggestion
TEST(SM89_CuTe_Ada, CooperativeGemm_e4m3e5m2f32_MMA) {
```

### thakkarV · 2025-03-21T14:55:33Z

https://github.com/NVIDIA/cutlass/pull/2177#discussion_r2007759054

```suggestion
TEST(SM89_CuTe_Ada, CooperativeGemm_e5m2e4m3f32_MMA) {
```

### thakkarV · 2025-03-21T14:55:38Z

https://github.com/NVIDIA/cutlass/pull/2177#discussion_r2007759204

```suggestion
TEST(SM89_CuTe_Ada, CooperativeGemm_e5m2e5m2f32_MMA) {
```

### kf-zhang · 2025-03-22T07:23:05Z

https://github.com/NVIDIA/cutlass/pull/2177#discussion_r2008701071

Copyright headers have been added to the new files. Please review and confirm.

### kf-zhang · 2025-03-22T07:29:05Z

https://github.com/NVIDIA/cutlass/pull/2177#discussion_r2008701984

✅ Done

### kf-zhang · 2025-03-22T07:29:21Z

https://github.com/NVIDIA/cutlass/pull/2177#discussion_r2008702021

✅ Done

### kf-zhang · 2025-03-22T07:29:28Z

https://github.com/NVIDIA/cutlass/pull/2177#discussion_r2008702028

✅ Done

### kf-zhang · 2025-03-22T07:29:38Z

https://github.com/NVIDIA/cutlass/pull/2177#discussion_r2008702044

✅ Done

### kf-zhang · 2025-03-22T07:29:47Z

https://github.com/NVIDIA/cutlass/pull/2177#discussion_r2008702058

✅ Done

### kf-zhang · 2025-03-22T07:29:55Z

https://github.com/NVIDIA/cutlass/pull/2177#discussion_r2008702068

✅ Done

### kf-zhang · 2025-03-22T07:30:03Z

https://github.com/NVIDIA/cutlass/pull/2177#discussion_r2008702098

✅ Done
