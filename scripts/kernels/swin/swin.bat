@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0..\..\.."
set "BUILD_DIR=%ROOT%\build"
set "CONFIG=Release"
set "PYTHON=python"

rem Some Codex/PowerShell sessions expose both PATH and Path. MSBuild treats
rem them as duplicate environment keys, so normalize to one spelling in this
rem child process while preserving the search path.
set "TINY_CUTLASS_ORIGINAL_PATH=%PATH%"
set "PATH="
set "Path=%TINY_CUTLASS_ORIGINAL_PATH%"

cmake -S "%ROOT%" -B "%BUILD_DIR%" -DTINY_CUTLASS_BUILD_FLASH_ATTENTION=OFF -DTINY_CUTLASS_BUILD_SWIN=ON -DTINY_CUTLASS_BUILD_CONV_FUSED=OFF -DTINY_CUTLASS_BUILD_NATTEN=OFF
if errorlevel 1 exit /b 1

cmake --build "%BUILD_DIR%" --config "%CONFIG%" --target swin
if errorlevel 1 exit /b 1

%PYTHON% "%ROOT%\scripts\kernels\swin\verify.py" --build-dir "%BUILD_DIR%" --config "%CONFIG%"
if errorlevel 1 exit /b 1

if /I "%CUTLASS_PROFILE%"=="1" goto profile

%PYTHON% "%ROOT%\scripts\kernels\swin\bench.py" --build-dir "%BUILD_DIR%" --config "%CONFIG%" --batch-size 1
if errorlevel 1 exit /b 1

goto end

:profile
%PYTHON% "%ROOT%\scripts\kernels\swin\bench.py" --build-dir "%BUILD_DIR%" --config "%CONFIG%" --batch-size 1 --nsys
if errorlevel 1 exit /b 1

:end
endlocal
