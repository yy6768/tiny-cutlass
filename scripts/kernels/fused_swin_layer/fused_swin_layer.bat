@echo off
setlocal EnableExtensions
set "ROOT=%~dp0..\..\.."
if not defined PYTHON set "PYTHON=python"
if not defined CUTLASS_ARCH set "CUTLASS_ARCH=89"
set "BUILD_DIR=%ROOT%\build\fused_swin_layer"
set "PYTHONDONTWRITEBYTECODE=1"
"%PYTHON%" -B "%ROOT%\csrc\tests\fused_swin_layer\verify.py" --source-only
if errorlevel 1 exit /b 1
cmake -S "%ROOT%\csrc\swin\fused_swin_layer" -B "%BUILD_DIR%" -DCMAKE_CUDA_ARCHITECTURES=%CUTLASS_ARCH%
if errorlevel 1 exit /b 1
cmake --build "%BUILD_DIR%" --config Release --target fused_swin_layer --parallel 2
if errorlevel 1 exit /b 1
"%PYTHON%" -B "%ROOT%\csrc\tests\fused_swin_layer\verify.py" --build-dir "%BUILD_DIR%"
if errorlevel 1 exit /b 1
"%PYTHON%" -B "%ROOT%\csrc\tests\fused_swin_layer\bench.py" --build-dir "%BUILD_DIR%"
if errorlevel 1 exit /b 1
endlocal
