@echo off
setlocal EnableExtensions
set "ROOT=%~dp0..\..\.."
set "BUILD_DIR=%ROOT%\build\natten"
set "EXE=%BUILD_DIR%\csrc\natten\Release\natten_fna.exe"
set "CUDA_TOOLKIT_DIR=%CUDA_PATH%"
if not defined CUDA_TOOLKIT_DIR set "CUDA_TOOLKIT_DIR=%CUDA_PATH_V12_9%"
if not defined CUDA_TOOLKIT_DIR set "CUDA_TOOLKIT_DIR=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
if not exist "%CUDA_TOOLKIT_DIR%\bin\nvcc.exe" (
  echo CUDA toolkit not found. Set CUDA_PATH.
  exit /b 1
)
if not "%CUDA_TOOLKIT_DIR:~-1%"=="\" set "CUDA_TOOLKIT_DIR=%CUDA_TOOLKIT_DIR%\"
set "CUDA_TOOLKIT_DIR_ARG=%CUDA_TOOLKIT_DIR:\=/%"
if not defined TINY_CUTLASS_CUDA_ARCH set "TINY_CUTLASS_CUDA_ARCH=89"

echo [natten] build
cmake -S "%ROOT%" -B "%BUILD_DIR%" -G "Visual Studio 17 2022" -A x64 "-DCMAKE_CUDA_ARCHITECTURES=%TINY_CUTLASS_CUDA_ARCH%" "-DCUDAToolkit_ROOT=%CUDA_TOOLKIT_DIR_ARG%" -DTINY_CUTLASS_BUILD_FLASH_ATTENTION=OFF -DTINY_CUTLASS_BUILD_SWIN=OFF -DTINY_CUTLASS_BUILD_CONV_FUSED=OFF -DTINY_CUTLASS_BUILD_NATTEN=ON -DTINY_CUTLASS_BUILD_CATMULL_ROM=OFF -DTINY_CUTLASS_BUILD_EVT=OFF
if errorlevel 1 exit /b 1
cmake --build "%BUILD_DIR%" --config Release --target natten_fna -- "/p:CudaToolkitDir=%CUDA_TOOLKIT_DIR_ARG%"
if errorlevel 1 exit /b 1

echo [natten] verify
python -B "%ROOT%\csrc\tests\natten\verify.py" --exe "%EXE%" --report "%BUILD_DIR%\verify.txt"
if errorlevel 1 exit /b 1

echo [natten] bench
python -B "%ROOT%\csrc\tests\natten\bench.py" --exe "%EXE%" --report "%BUILD_DIR%\bench.txt"
if errorlevel 1 exit /b 1
endlocal
