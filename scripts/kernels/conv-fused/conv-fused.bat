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

cmake --build "%BUILD_DIR%" --target conv_fused conv_pool conv_relu_pool conv_relu_pool_plugin conv_relu_pool_trt conv1x1_relu_conv1x1_relu conv1x1_relu_conv1x1_relu_trt
if errorlevel 1 exit /b 1

"%BUILD_DIR%\tests\conv-fused\conv_pool.exe"
if errorlevel 1 exit /b 1

"%BUILD_DIR%\tests\conv-fused\conv_relu_pool.exe"
if errorlevel 1 exit /b 1

"%BUILD_DIR%\tests\conv-fused\conv_relu_pool_plugin.exe"
if errorlevel 1 exit /b 1

"%BUILD_DIR%\tests\conv-fused\conv1x1_relu_conv1x1_relu.exe"
if errorlevel 1 exit /b 1

if /I "%CONV_FUSED_SKIP_BENCH%"=="1" goto done

set "ARTIFACT_DIR=%BUILD_DIR%\reference"

python "%ROOT%\scripts\kernels\conv-fused\export_conv_relu_pool_reference.py" --output-dir "%ARTIFACT_DIR%"
if errorlevel 1 exit /b 1

python "%ROOT%\scripts\kernels\conv-fused\bench.py" --artifact-dir "%ARTIFACT_DIR%" --build-dir "%BUILD_DIR%"
if errorlevel 1 exit /b 1

python "%ROOT%\scripts\kernels\conv-fused\verify.py" --artifact-dir "%ARTIFACT_DIR%"
if errorlevel 1 exit /b 1

if /I "%CONV_FUSED_SKIP_FP8%"=="1" goto done

set "FP8_ARTIFACT_DIR=%BUILD_DIR%\fp8-reference"

python "%ROOT%\scripts\kernels\conv-fused\export_fp8_reference.py" --output-dir "%FP8_ARTIFACT_DIR%"
if errorlevel 1 exit /b 1

python "%ROOT%\scripts\kernels\conv-fused\bench.py" --mode fp8 --artifact-dir "%FP8_ARTIFACT_DIR%" --build-dir "%BUILD_DIR%"
if errorlevel 1 exit /b 1

python "%ROOT%\scripts\kernels\conv-fused\verify.py" --mode fp8 --artifact-dir "%FP8_ARTIFACT_DIR%"
if errorlevel 1 exit /b 1

:done
endlocal
