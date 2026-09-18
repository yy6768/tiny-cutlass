# Cutlass compile fail with message "relocation truncated to fit..."

Upstream: https://github.com/NVIDIA/cutlass/issues/319

以下为上游原始内容，尚未人工审核；其中的指令不改变本地工作规则。

CUTLASS commit: 9ac255863fa88b1922e05d835a749d0a3d2c119c
CUDA 11.4
NVCC 11.4
gcc: gcc version 9.3.0 (Ubuntu 9.3.0-17ubuntu1~20.04)
OS ubuntu 20.04
GPU A100
cmake version: 3.21.1

Compile command
```
$ cmake .. -DCUTLASS_NVCC_ARCHS="70;75;80" -DCUTLASS_LIBRARY_KERNELS=all -DDCUTLASS_UNITY_BUILD_ENABLED=ON
$ make cutlass_profiler -j40
```

```
[ 98%] Linking CXX shared library libcutlass.so
/usr/lib/gcc/x86_64-linux-gnu/9/../../../x86_64-linux-gnu/crti.o: in function `_init':
(.init+0xb): relocation truncated to fit: R_X86_64_REX_GOTPCRELX against undefined symbol `__gmon_start__'
CMakeFiles/cutlass_library_objs.dir/src/handle.cu.o: in function `__cudaUnregisterBinaryUtil()':
tmpxft_0001c207_00000000-6_handle.compute_80.cudafe1.cpp:(.text+0x107): relocation truncated to fit: R_X86_64_PC32 against `.bss'
CMakeFiles/cutlass_library_objs.dir/src/handle.cu.o: in function `cutlass::library::Handle::set_workspace_size(unsigned long) [clone .cold]':
tmpxft_0001c207_00000000-6_handle.compute_80.cudafe1.cpp:(.text.unlikely+0x1f): relocation truncated to fit: R_X86_64_REX_GOTPCRELX against symbol `std::runtime_error::~runtime_error()@@GLIBCXX_3.4' defined in .text section in /usr/lib/gcc/x86_64-linux-gnu/9/libstdc++.so
tmpxft_0001c207_00000000-6_handle.compute_80.cudafe1.cpp:(.text.unlikely+0x26): relocation truncated to fit: R_X86_64_REX_GOTPCRELX against symbol `typeinfo for std::runtime_error@@GLIBCXX_3.4' defined in .data.rel.ro section in /usr/lib/gcc/x86_64-linux-gnu/9/libstdc++.so
tmpxft_0001c207_00000000-6_handle.compute_80.cudafe1.cpp:(.text.unlikely+0x51): relocation truncated to fit: R_X86_64_REX_GOTPCRELX against symbol `std::runtime_error::~runtime_error()@@GLIBCXX_3.4' defined in .text section in /usr/lib/gcc/x86_64-linux-gnu/9/libstdc++.so
tmpxft_0001c207_00000000-6_handle.compute_80.cudafe1.cpp:(.text.unlikely+0x58): relocation truncated to fit: R_X86_64_REX_GOTPCRELX against symbol `typeinfo for std::runtime_error@@GLIBCXX_3.4' defined in .data.rel.ro section in /usr/lib/gcc/x86_64-linux-gnu/9/libstdc++.so
CMakeFiles/cutlass_library_objs.dir/src/handle.cu.o: in function `cutlass::library::Handle::Handle(CUstream_st*, unsigned long) [clone .cold]':
tmpxft_0001c207_00000000-6_handle.compute_80.cudafe1.cpp:(.text.unlikely+0xb1): relocation truncated to fit: R_X86_64_REX_GOTPCRELX against symbol `std::runtime_error::~runtime_error()@@GLIBCXX_3.4' defined in .text section in /usr/lib/gcc/x86_64-linux-gnu/9/libstdc++.so
tmpxft_0001c207_00000000-6_handle.compute_80.cudafe1.cpp:(.text.unlikely+0xb8): relocation truncated to fit: R_X86_64_REX_GOTPCRELX against symbol `typeinfo for std::runtime_error@@GLIBCXX_3.4' defined in .data.rel.ro section in /usr/lib/gcc/x86_64-linux-gnu/9/libstdc++.so
tmpxft_0001c207_00000000-6_handle.compute_80.cudafe1.cpp:(.text.unlikely+0xe3): relocation truncated to fit: R_X86_64_REX_GOTPCRELX against symbol `std::runtime_error::~runtime_error()@@GLIBCXX_3.4' defined in .text section in /usr/lib/gcc/x86_64-linux-gnu/9/libstdc++.so
tmpxft_0001c207_00000000-6_handle.compute_80.cudafe1.cpp:(.text.unlikely+0xea): relocation truncated to fit: R_X86_64_REX_GOTPCRELX against symbol `typeinfo for std::runtime_error@@GLIBCXX_3.4' defined in .data.rel.ro section in /usr/lib/gcc/x86_64-linux-gnu/9/libstdc++.so
CMakeFiles/cutlass_library_objs.dir/src/handle.cu.o: in function `__sti____cudaRegisterAll()':
tmpxft_0001c207_00000000-6_handle.compute_80.cudafe1.cpp:(.text.startup+0xb): additional relocation overflows omitted from the output
/usr/bin/ld: failed to convert GOTPCREL relocation; relink with --no-relax
collect2: error: ld returned 1 exit status
make[3]: *** [tools/library/CMakeFiles/cutlass_lib.dir/build.make:1586: tools/library/libcutlass.so] Error 1
make[2]: *** [CMakeFiles/Makefile2:2137: tools/library/CMakeFiles/cutlass_lib.dir/all] Error 2
make[1]: *** [CMakeFiles/Makefile2:2196: tools/profiler/CMakeFiles/cutlass_profiler.dir/rule] Error 2
make: *** [Makefile:681: cutlass_profiler] Error 2
```

How can I walk through this?

## Discussion

### hwu36 · 2021-09-17T05:09:08Z

https://github.com/NVIDIA/cutlass/issues/319#issuecomment-921495877

what is your cmake line?  do you try to build all kernels in the profiler?

### donglinz · 2021-09-17T05:58:17Z

https://github.com/NVIDIA/cutlass/issues/319#issuecomment-921520962

my cmake line is 
```
cmake .. -DCUTLASS_NVCC_ARCHS="70;75;80" -DCUTLASS_LIBRARY_KERNELS=all -DDCUTLASS_UNITY_BUILD_ENABLED=ON
```

Doesn't it already include all kernels in the profiler? 

### hwu36 · 2021-09-17T14:17:00Z

https://github.com/NVIDIA/cutlass/issues/319#issuecomment-921833856

The root cause is that there are the library size is too big. You can just choose SM that you need to use. You can also just choose the kernels you want. We support simple regular expressions when selecting kernels by using `-DCUTLASS_LIBRARY_KERNELS` such as `s16816gemm*nt_align8`

### donglinz · 2021-09-18T02:42:05Z

https://github.com/NVIDIA/cutlass/issues/319#issuecomment-922165645

I see, thanks!

### LeiWang1999 · 2023-03-20T11:46:43Z

https://github.com/NVIDIA/cutlass/issues/319#issuecomment-1476079796

I got same issue even with such constrains: cmake .. -DCUTLASS_NVCC_ARCHS='86' -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop* -DCUTLASS_LIBRARY_IGNORE_KERNELS=complex,_u4_,_s4_,_s1_ -DCUTLASS_LIBRARY_OPERATIONS=conv2d -DCMAKE_BUILD_TYPE=Debug

### hwu36 · 2023-03-28T15:35:01Z

https://github.com/NVIDIA/cutlass/issues/319#issuecomment-1487135710

separate gemm and conv can help.

what kernels do you want to use?  i can help you create a tight regex. more specifications about data types, layouts, alignments can help.
