@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ---------------------------------------------------------------------------
rem conv-quant-bound: measure the achievable-performance ceiling of int8 vs fp16
rem conv2d fprop on SM89, using cutlass_profiler as an exhaustive tile/stage
rem sweep. The profiler's own reference check is the verify gate: any kernel
rem whose Status is not "Passed" is excluded from the bench conclusion.
rem
rem Workflow order is build -> verify -> bench, per the cutlass-kernel skill.
rem ---------------------------------------------------------------------------

set "ROOT=%~dp0..\..\.."
set "BUILD_DIR=%ROOT%\build\conv-quant-bound"
set "OUT_DIR=%ROOT%\profiling\csv\conv-quant-bound"
set "CONFIG=Release"

rem Kernel filters. Three comparable groups over the same NHWC problem:
rem   s8_i16832  int8 in / int8 out, i32 accumulate, requantize in the epilogue
rem   f16_s16816 fp16 in / fp16 out, f32 accumulate  (numerically safe fp16)
rem   h16816     fp16 in / fp16 out, f16 accumulate  (fastest fp16)
set "KERNELS=cutlass_tensorop_s8_i16832fprop_optimized_s8_*_nhwc_align16,cutlass_tensorop_f16_s16816fprop_optimized_f16_*_nhwc_align8,cutlass_tensorop_h16816fprop_optimized_*_nhwc_align8"

rem Problem: N=1 56x56 C=256 -> K=256, 3x3, pad 1, stride 1. Compute-bound,
rem C and K both multiples of 16 so int8 NHWC alignment holds.
set "PROBLEM=--n=1 --h=56 --w=56 --c=256 --k=256 --r=3 --s=3 --pad_h=1 --pad_w=1 --stride_h=1 --stride_w=1 --dilation_h=1 --dilation_w=1"

set "VSDEVCMD=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if exist "%VSDEVCMD%" (
  call "%VSDEVCMD%" -arch=x64 -host_arch=x64
  if errorlevel 1 exit /b 1
)

rem --- build ------------------------------------------------------------------
cmake -S "%ROOT%\3rdparty\cutlass" -B "%BUILD_DIR%" -G Ninja ^
  -DCMAKE_BUILD_TYPE=%CONFIG% ^
  -DCUTLASS_NVCC_ARCHS=89 ^
  -DCUTLASS_ENABLE_TESTS=OFF ^
  -DCUTLASS_ENABLE_EXAMPLES=OFF ^
  -DCUTLASS_UNITY_BUILD_ENABLED=ON ^
  -DCUTLASS_LIBRARY_OPERATIONS=conv2d ^
  -DCUTLASS_LIBRARY_KERNELS=%KERNELS%
if errorlevel 1 exit /b 1

cmake --build "%BUILD_DIR%" --target cutlass_profiler
if errorlevel 1 exit /b 1

set "PROFILER=%BUILD_DIR%\tools\profiler\cutlass_profiler.exe"
if not exist "%PROFILER%" (
  echo ERROR: cutlass_profiler not found at %PROFILER%
  exit /b 1
)

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

rem --- verify -----------------------------------------------------------------
rem One profiling pass with reference verification enabled. The device reference
rem covers both the i32-accumulate int8 path and the fp16 paths. profiler_status
rem is non-zero if any kernel fails verification.
echo.
echo === verify: reference parity ===
"%PROFILER%" --operation=Conv2d --conv_kind=fprop %PROBLEM% ^
  --providers=cutlass --verification-enabled=true --verification-providers=device ^
  --profiling-iterations=1 --warmup-iterations=1 ^
  --output="%OUT_DIR%\verify.csv"
if errorlevel 1 (
  echo ERROR: profiler returned non-zero during verification.
  exit /b 1
)

rem --- bench ------------------------------------------------------------------
rem Verification stays on so every timed row carries its own parity status; the
rem analysis step drops any row whose Status is not "Passed".
echo.
echo === bench: tile/stage sweep ===
"%PROFILER%" --operation=Conv2d --conv_kind=fprop %PROBLEM% ^
  --providers=cutlass --verification-enabled=true --verification-providers=device ^
  --warmup-iterations=100 --profiling-iterations=1000 ^
  --output="%OUT_DIR%\bench.csv"
if errorlevel 1 (
  echo ERROR: profiler returned non-zero during benchmark.
  exit /b 1
)

echo.
echo CSV artifacts written under %OUT_DIR%
endlocal
