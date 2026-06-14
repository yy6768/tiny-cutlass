# FlashAttention

这个目录是 tiny-cutlass 的 FlashAttention 学习工作区。kernel 通过
`csrc/tests/flash-attention` 下的共享 harness 构建、验证和 benchmark。

## 范围

- 当前优化重点：SM80 FP16 和 SM89 FP8。
- reference backend：cuDNN frontend SDPA。
- 不提供 CPU reference fallback。
- 标准流程：先 build，再用 cuDNN verify，通过后才 benchmark/profile。

## 依赖

- 启用 MSVC 的 CUDA Toolkit。
- `3rdparty/cutlass` 子模块。
- 本机安装 cuDNN backend。
- `3rdparty/cudnn-frontend`。

参考资料：

- NVIDIA cuDNN 安装文档：<https://docs.nvidia.com/deeplearning/cudnn/installation/>
- NVIDIA cuDNN frontend：<https://github.com/NVIDIA/cudnn-frontend>

## 预期目录布局

`cmake/CUDNN.cmake` 使用一个明确的 cuDNN 路径和一个 CUDA 版本，不搜索 fallback
位置。

设置 `CUDNN_PATH` 指向 cuDNN 安装目录：

```powershell
$env:CUDNN_PATH = "C:\Program Files\NVIDIA\CUDNN\v9.21"
```

CUDA 12.9 下预期 backend 布局：

```text
%CUDNN_PATH%\include\12.9\cudnn.h
%CUDNN_PATH%\lib\12.9\x64\cudnn.lib
%CUDNN_PATH%\bin\12.9\x64\cudnn64_9.dll
```

预期 frontend 布局：

```text
3rdparty\cudnn-frontend\include\cudnn_frontend.h
```

如果 cuDNN 放在非标准位置，直接传入路径：

```powershell
-DTINY_CUTLASS_CUDNN_PATH:PATH="D:\path\to\cudnn"
```

如果 cuDNN package 子目录和 CUDA compiler 版本不一致，显式传入版本：

```powershell
-DTINY_CUTLASS_CUDNN_CUDA_VERSION:STRING=12.9
```

## 配置

当前 CUDA 12.9 + cuDNN v9.21 配置：

```powershell
$env:CUDNN_PATH = "C:\Program Files\NVIDIA\CUDNN\v9.21"

cmake -S . -B build `
  -DTINY_CUTLASS_BUILD_FLASH_ATTENTION=ON `
  -DTINY_CUTLASS_BUILD_SWIN=OFF
```

如果 CMake 仍然使用不同 CUDA compiler，固定 cuDNN CUDA 版本：

```powershell
cmake -S . -B build `
  -DTINY_CUTLASS_BUILD_FLASH_ATTENTION=ON `
  -DTINY_CUTLASS_BUILD_SWIN=OFF `
  -DTINY_CUTLASS_CUDNN_CUDA_VERSION:STRING=12.9
```

FlashAttention CMake target 会把需要的 cuDNN 和 CUDA runtime DLL 复制到 `build/`
下的 executable 输出目录，所以运行测试时不需要把 cuDNN 或 CUDA `bin` 加到全局
`PATH`。

## 诊断

```powershell
Test-Path "$env:CUDNN_PATH\include\12.9\cudnn.h"
Test-Path "$env:CUDNN_PATH\lib\12.9\x64\cudnn.lib"
Test-Path "$env:CUDNN_PATH\bin\12.9\x64\cudnn64_9.dll"
Test-Path "3rdparty\cudnn-frontend\include\cudnn_frontend.h"
```

检查 cuDNN 版本：

```powershell
Select-String -Path "$env:CUDNN_PATH\include\12.9\cudnn_version.h" `
  -Pattern "CUDNN_MAJOR|CUDNN_MINOR|CUDNN_PATCHLEVEL"
```

## 构建

```powershell
cmake --build build --config Release --target flash_attention_test
```

兼容 target：

```powershell
cmake --build build --config Release --target `
  flash_attention_00_naive_attention_test `
  flash_attention_01_online_softmax_attention_test `
  flash_attention_02_tiled_online_attention_test
```

## 验证

列出 kernel：

```powershell
build\csrc\flash-attention\Release\flash_attention_test.exe --kernel=list
```

运行一个小型 cuDNN verified case：

```powershell
build\csrc\flash-attention\Release\flash_attention_test.exe `
  --kernel=00-naive `
  --batch_size=1 `
  --head_number=1 `
  --seq_length=64 `
  --seq_length_kv=64 `
  --head_size=64 `
  --head_size_v=64 `
  --iterations=1 `
  --reference=cudnn `
  --reference-check=true
```

随后用相同 shape 验证 `01-online-softmax` 和 `02-tiled-online`。

## 性能约束

- 没有通过 cuDNN reference parity 的结果不能作为性能结论。
- benchmark 和 profiling artifact 必须落在 `build/`。
- 每个性能数字都应该记录 kernel、shape、dtype、GPU、编译 arch 和误差阈值。
