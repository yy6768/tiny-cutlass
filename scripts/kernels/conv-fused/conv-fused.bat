@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0..\..\.."
set "BUILD_DIR=%ROOT%\build\conv-fused"
set "CONFIG=Release"
set "VSDEVCMD=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
set "TENSORRT_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT"
set "PATH=%TENSORRT_ROOT%\lib;%PATH%"

if exist "%VSDEVCMD%" (
  call "%VSDEVCMD%" -arch=x64 -host_arch=x64
  if errorlevel 1 exit /b 1
)

cmake -S "%ROOT%" -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=%CONFIG% -DCMAKE_CUDA_ARCHITECTURES=89 -DTINY_CUTLASS_BUILD_FLASH_ATTENTION=OFF -DTINY_CUTLASS_BUILD_CONV_FUSED=ON
if errorlevel 1 exit /b 1

cmake --build "%BUILD_DIR%" --target conv_fused conv_b2b conv_b2b_relu conv_b2b_trt
if errorlevel 1 exit /b 1

"%BUILD_DIR%\tests\conv-fused\conv_b2b.exe"
if errorlevel 1 exit /b 1

"%BUILD_DIR%\tests\conv-fused\conv_b2b_relu.exe"
if errorlevel 1 exit /b 1

if /I "%CONV_FUSED_SKIP_BENCH%"=="1" goto done

if /I "%CONV_FUSED_SKIP_FP8%"=="1" goto done

set "FP8_ARTIFACT_DIR=%BUILD_DIR%\fp8-reference"

python "%ROOT%\scripts\kernels\conv-fused\export_fp8_reference.py" --output-dir "%FP8_ARTIFACT_DIR%"
if errorlevel 1 exit /b 1

"%BUILD_DIR%\tests\conv-fused\conv_b2b_trt.exe" --artifact-dir "%FP8_ARTIFACT_DIR%"
if errorlevel 1 exit /b 1

python "%ROOT%\scripts\kernels\conv-fused\verify.py" --mode fp8 --artifact-dir "%FP8_ARTIFACT_DIR%"
if errorlevel 1 exit /b 1

python "%ROOT%\scripts\kernels\conv-fused\bench.py" --mode fp8 --artifact-dir "%FP8_ARTIFACT_DIR%" --build-dir "%BUILD_DIR%"
if errorlevel 1 exit /b 1

:done
endlocal
