@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0..\..\.."
set "BUILD_DIR=%ROOT%\build"
set "CONFIG=Release"
set "TARGET=natten_fna"
set "EXE=%BUILD_DIR%\csrc\natten\%CONFIG%\%TARGET%.exe"
set "CUDA_TOOLKIT_DIR=%CUDA_PATH%"

if not "%CUDA_TOOLKIT_DIR%"=="" goto cuda_found
if "%CUDA_PATH_V12_9%"=="" goto try_programw6432
set "CUDA_TOOLKIT_DIR=%CUDA_PATH_V12_9%"
goto cuda_found

:try_programw6432
if "%ProgramW6432%"=="" goto try_programfiles
if not exist "%ProgramW6432%\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" goto try_programfiles
set "CUDA_TOOLKIT_DIR=%ProgramW6432%\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
goto cuda_found

:try_programfiles
if "%ProgramFiles%"=="" goto cuda_missing
if not exist "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" goto cuda_missing
set "CUDA_TOOLKIT_DIR=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
goto cuda_found

:cuda_missing
echo CUDA toolkit not found. Set CUDA_PATH or install CUDA Toolkit.
exit /b 1

:cuda_found
if not "%CUDA_TOOLKIT_DIR:~-1%"=="\" set "CUDA_TOOLKIT_DIR=%CUDA_TOOLKIT_DIR%\"
set "CUDA_TOOLKIT_DIR_ARG=%CUDA_TOOLKIT_DIR:\=/%"

echo [natten] build
cmake -S "%ROOT%" -B "%BUILD_DIR%" "-DCUDAToolkit_ROOT=%CUDA_TOOLKIT_DIR_ARG%" -DTINY_CUTLASS_BUILD_FLASH_ATTENTION=OFF -DTINY_CUTLASS_BUILD_SWIN=OFF -DTINY_CUTLASS_BUILD_CONV_FUSED=OFF -DTINY_CUTLASS_BUILD_NATTEN=ON
if errorlevel 1 exit /b 1

cmake --build "%BUILD_DIR%" --config "%CONFIG%" --target "%TARGET%" -- "/p:CudaToolkitDir=%CUDA_TOOLKIT_DIR_ARG%"
if errorlevel 1 exit /b 1

echo [natten] verify
"%EXE%" --batch_size=2 --length=128 --heads=2 --head_dim=32 --head_dim_value=32 --kernel_size=33
if errorlevel 1 exit /b 1

echo [natten] bench
echo NATTEN FNA has no launchable kernel; benchmark skipped.

endlocal
