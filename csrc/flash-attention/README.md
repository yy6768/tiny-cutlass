# FlashAttention

This folder is the tiny-cutlass FlashAttention study workspace. Kernels are built and verified through the shared harness under `csrc/tests/flash-attention`.

## Scope

- Current optimization focus: SM80 FP16 and SM89 FP8.
- Reference backend: cuDNN frontend SDPA.
- No CPU reference fallback.
- Normal workflow: build, verify against cuDNN, then benchmark/profile.

## Dependencies

- CUDA Toolkit with MSVC enabled.
- CUTLASS submodule at `3rdparty/cutlass`.
- cuDNN backend installed on the machine.
- cuDNN frontend at `3rdparty/cudnn-frontend`.

References:

- NVIDIA cuDNN install docs: <https://docs.nvidia.com/deeplearning/cudnn/installation/>
- NVIDIA cuDNN frontend: <https://github.com/NVIDIA/cudnn-frontend>

## Expected Layout

`cmake/CUDNN.cmake` uses one explicit cuDNN path and one CUDA version. It does not search fallback locations.

Set `CUDNN_PATH` to the cuDNN installation path:

```powershell
$env:CUDNN_PATH = "C:\Program Files\NVIDIA\CUDNN\v9.21"
```

With CUDA 12.9, the expected backend layout is:

```text
%CUDNN_PATH%\include\12.9\cudnn.h
%CUDNN_PATH%\lib\12.9\x64\cudnn.lib
%CUDNN_PATH%\bin\12.9\x64\cudnn64_9.dll
```

The expected frontend layout is:

```text
3rdparty\cudnn-frontend\include\cudnn_frontend.h
```

If you keep cuDNN in a nonstandard location, pass the path directly:

```powershell
-DTINY_CUTLASS_CUDNN_PATH:PATH="D:\path\to\cudnn"
```

If the cuDNN package subdirectory does not match the CUDA compiler version, pass the version explicitly:

```powershell
-DTINY_CUTLASS_CUDNN_CUDA_VERSION:STRING=12.9
```

## Configure

For the current CUDA 12.9 + cuDNN v9.21 setup:

```powershell
$env:CUDNN_PATH = "C:\Program Files\NVIDIA\CUDNN\v9.21"

cmake -S . -B build `
  -DTINY_CUTLASS_BUILD_FLASH_ATTENTION=ON `
  -DTINY_CUTLASS_BUILD_SWIN=OFF
```

If CMake is still using a different CUDA compiler, pin the cuDNN CUDA version:

```powershell
cmake -S . -B build `
  -DTINY_CUTLASS_BUILD_FLASH_ATTENTION=ON `
  -DTINY_CUTLASS_BUILD_SWIN=OFF `
  -DTINY_CUTLASS_CUDNN_CUDA_VERSION:STRING=12.9
```

The flash-attention CMake target copies the required cuDNN and CUDA runtime DLLs
into the executable output directories under `build/`, so test execution does
not require adding cuDNN or CUDA `bin` directories to the global `PATH`.

## Diagnose

```powershell
Test-Path "$env:CUDNN_PATH\include\12.9\cudnn.h"
Test-Path "$env:CUDNN_PATH\lib\12.9\x64\cudnn.lib"
Test-Path "$env:CUDNN_PATH\bin\12.9\x64\cudnn64_9.dll"
Test-Path "3rdparty\cudnn-frontend\include\cudnn_frontend.h"
```

Check the cuDNN version:

```powershell
Select-String -Path "$env:CUDNN_PATH\include\12.9\cudnn_version.h" `
  -Pattern "CUDNN_MAJOR|CUDNN_MINOR|CUDNN_PATCHLEVEL"
```

## Build

```powershell
cmake --build build --config Release --target flash_attention_test
```

Compatibility targets:

```powershell
cmake --build build --config Release --target `
  flash_attention_00_naive_attention_test `
  flash_attention_01_online_softmax_attention_test `
  flash_attention_02_tiled_online_attention_test
```

## Verify

List kernels:

```powershell
build\csrc\flash-attention\Release\flash_attention_test.exe --kernel=list
```

Run a small cuDNN-verified case:

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

Then run the same shape for `01-online-softmax` and `02-tiled-online`.
