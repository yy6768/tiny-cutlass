@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0..\..\.."
set "BUILD_DIR=%ROOT%\build\swin"
set "CONFIG=Release"
set "PYTHON=python"

rem Normalize PATH spelling because MSBuild rejects duplicate PATH/Path keys.
set "TINY_CUTLASS_ORIGINAL_PATH=%PATH%"
set "PATH="
set "Path=%TINY_CUTLASS_ORIGINAL_PATH%"

cmake -S "%ROOT%" -B "%BUILD_DIR%" -DCMAKE_CUDA_ARCHITECTURES=89 -DTINY_CUTLASS_BUILD_FLASH_ATTENTION=OFF -DTINY_CUTLASS_BUILD_SWIN=ON -DTINY_CUTLASS_BUILD_CONV_FUSED=OFF -DTINY_CUTLASS_BUILD_NATTEN=OFF
if errorlevel 1 exit /b 1

cmake --build "%BUILD_DIR%" --config "%CONFIG%" --target swin
if errorlevel 1 exit /b 1

%PYTHON% "%ROOT%\scripts\kernels\swin\verify.py" --build-dir "%BUILD_DIR%" --config "%CONFIG%" --official --checkpoint-dir "%ROOT%\checkpoint"
if errorlevel 1 exit /b 1

set "PROFILE_FLAGS="
if /I "%CUTLASS_PROFILE%"=="1" set "PROFILE_FLAGS=--official --nsys --ncu"

%PYTHON% "%ROOT%\scripts\kernels\swin\bench.py" --build-dir "%BUILD_DIR%" --config "%CONFIG%" --batch-size 1 --checkpoint-dir "%ROOT%\checkpoint" --report-dir "%ROOT%\profile\swin" %PROFILE_FLAGS%
if errorlevel 1 exit /b 1

endlocal
