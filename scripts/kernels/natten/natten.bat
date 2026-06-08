@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0..\..\.."
set "BUILD_DIR=%ROOT%\build"
set "CONFIG=Release"
set "TARGET=natten_fna_test"
set "EXE=%BUILD_DIR%\csrc\natten\%CONFIG%\%TARGET%.exe"
set "CUDA_TOOLKIT_DIR=%CUDA_PATH%"

if exist "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" (
  set "CUDA_TOOLKIT_DIR=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
)

if "%CUDA_TOOLKIT_DIR%"=="" (
  echo CUDA toolkit not found. Set CUDA_PATH or install CUDA Toolkit.
  exit /b 1
)

if not "%CUDA_TOOLKIT_DIR:~-1%"=="\" set "CUDA_TOOLKIT_DIR=%CUDA_TOOLKIT_DIR%\"
set "CUDA_TOOLKIT_DIR_ARG=%CUDA_TOOLKIT_DIR:\=/%"

cmake -S "%ROOT%" -B "%BUILD_DIR%" "-DCUDAToolkit_ROOT=%CUDA_TOOLKIT_DIR_ARG%" -DTINY_CUTLASS_BUILD_FLASH_ATTENTION=OFF -DTINY_CUTLASS_BUILD_SWIN=OFF -DTINY_CUTLASS_BUILD_CONV_FUSED=OFF -DTINY_CUTLASS_BUILD_NATTEN=ON
if errorlevel 1 exit /b 1

cmake --build "%BUILD_DIR%" --config "%CONFIG%" --target "%TARGET%" -- "/p:CudaToolkitDir=%CUDA_TOOLKIT_DIR_ARG%"
if errorlevel 1 exit /b 1

"%EXE%" --batch_size=2 --length=128 --heads=2 --head_dim=32 --head_dim_value=32 --kernel_size=33
if errorlevel 1 exit /b 1

if /I "%CUTLASS_PROFILE%"=="1" goto profile
goto end

:profile
echo NATTEN FNA currently has only an interface scaffold and no launchable kernel; profiling is disabled until a CUTLASS-native TensorOp implementation exists.
exit /b 1

set "REPORT_DIR=%BUILD_DIR%\reports\natten"
if not exist "!REPORT_DIR!" mkdir "!REPORT_DIR!"

set "NCU_BASE=!REPORT_DIR!\%TARGET%"
ncu --force-overwrite --set full --launch-skip 4 --launch-count 4 --page raw --csv --import-source=1 --import-sass=1 -o "!NCU_BASE!" "%EXE%" --batch_size=4 --length=1024 --heads=8 --head_dim=32 --head_dim_value=32 --kernel_size=129 --iterations=20 --reference-check=false > "!NCU_BASE!.csv"
if errorlevel 1 exit /b 1

set "NSYS_BASE=!REPORT_DIR!\%TARGET%"
nsys profile --force-overwrite=true --trace=cuda,nvtx -o "!NSYS_BASE!" "%EXE%" --batch_size=4 --length=1024 --heads=8 --head_dim=32 --head_dim_value=32 --kernel_size=129 --iterations=20 --reference-check=false
if errorlevel 1 exit /b 1

:end
endlocal
