@echo off
setlocal EnableExtensions
set "ROOT=%~dp0..\..\.."
set "BUILD_DIR=%ROOT%\build\swin"
if not defined PYTHON set "PYTHON=python"
if not defined CUTLASS_ARCH set "CUTLASS_ARCH=89"
set "PYTHONDONTWRITEBYTECODE=1"
set "TINY_CUTLASS_ORIGINAL_PATH=%PATH%"
set "PATH="
set "Path=%TINY_CUTLASS_ORIGINAL_PATH%"
set "BUILD_TARGET=swin"
set "VERIFY_FLAGS=--all-families"
set "BENCH_FLAGS="
if /I "%~1"=="grouped-gemm" (
  set "BUILD_TARGET=swin_grouped_gemm"
  set "VERIFY_FLAGS=--case swin_grouped_gemm"
  set "BENCH_FLAGS=--grouped-gemm"
)
if /I "%~1"=="block" (
  set "BUILD_TARGET=swin_block"
  set "VERIFY_FLAGS=--block"
  set "BENCH_FLAGS=--block"
)

"%PYTHON%" "%ROOT%\csrc\tests\swin\verify.py" --source-only
if errorlevel 1 exit /b 1

cmake -S "%ROOT%" -B "%BUILD_DIR%" -DCMAKE_CUDA_ARCHITECTURES=%CUTLASS_ARCH% ^
  -DTINY_CUTLASS_BUILD_SWIN=ON -DTINY_CUTLASS_BUILD_FLASH_ATTENTION=OFF ^
  -DTINY_CUTLASS_BUILD_CONV_FUSED=OFF -DTINY_CUTLASS_BUILD_NATTEN=OFF ^
  -DTINY_CUTLASS_BUILD_CATMULL_ROM=OFF -DTINY_CUTLASS_BUILD_EVT=OFF
if errorlevel 1 exit /b 1
cmake --build "%BUILD_DIR%" --config Release --target %BUILD_TARGET% --parallel
if errorlevel 1 exit /b 1
"%PYTHON%" "%ROOT%\csrc\tests\swin\verify.py" --build-dir "%BUILD_DIR%" %VERIFY_FLAGS%
if errorlevel 1 exit /b 1
set "PROFILE_FLAGS="
if /I "%CUTLASS_PROFILE%"=="1" set "PROFILE_FLAGS=--ncu --nsys"
"%PYTHON%" "%ROOT%\csrc\tests\swin\bench.py" --build-dir "%BUILD_DIR%" %BENCH_FLAGS% %PROFILE_FLAGS%
if errorlevel 1 exit /b 1
endlocal
