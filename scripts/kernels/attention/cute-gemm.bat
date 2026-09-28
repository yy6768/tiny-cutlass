@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem CuTe warm-up GEMM: build -> verify (cuBLAS) -> bench.
rem Set CUTLASS_PROFILE=1 to also collect Nsight reports under build\reports.

set "ROOT=%~dp0..\..\.."
set "BUILD_DIR=%ROOT%\build"
set "CONFIG=Release"
set "TARGET=cute_gemm"
set "EXE=%BUILD_DIR%\csrc\flash-attention\%CONFIG%\%TARGET%.exe"

set "VERIFY_ARGS=--m=256 --n=128 --k=64 --iterations=1 --reference-check=true"
set "BENCH_ARGS=--m=4096 --n=4096 --k=4096 --iterations=20 --reference-check=false"

cmake -S "%ROOT%" -B "%BUILD_DIR%" -DTINY_CUTLASS_BUILD_FLASH_ATTENTION=ON -DTINY_CUTLASS_BUILD_CONV_FUSED=OFF
if errorlevel 1 exit /b 1

cmake --build "%BUILD_DIR%" --config "%CONFIG%" --target "%TARGET%"
if errorlevel 1 exit /b 1

rem Small shape first: a layout or partition bug shows up here in a fraction of
rem the time, and with an MAE small enough to read.
"%EXE%" %VERIFY_ARGS%
if errorlevel 1 exit /b 1

rem beta != 0 exercises the epilogue's C read, which the default beta = 0 skips.
"%EXE%" --m=256 --n=128 --k=64 --alpha=1.5 --beta=0.5 --iterations=1 --reference-check=true
if errorlevel 1 exit /b 1

"%EXE%" %BENCH_ARGS%
if errorlevel 1 exit /b 1

if /I "%CUTLASS_PROFILE%"=="1" goto profile
goto end

:profile
set "REPORT_DIR=%BUILD_DIR%\reports\flash-attention\cute-gemm"
if not exist "!REPORT_DIR!" mkdir "!REPORT_DIR!"

set "NCU_BASE=!REPORT_DIR!\%TARGET%"
ncu --force-overwrite --set full --launch-skip 6 --launch-count 3 --page raw --csv --import-source=1 --import-sass=1 -o "!NCU_BASE!" "%EXE%" %BENCH_ARGS% > "!NCU_BASE!.csv"
if errorlevel 1 exit /b 1

set "NSYS_BASE=!REPORT_DIR!\%TARGET%"
nsys profile --force-overwrite=true --trace=cuda,nvtx -o "!NSYS_BASE!" "%EXE%" %BENCH_ARGS%
if errorlevel 1 exit /b 1

pushd "!REPORT_DIR!"
nsys stats --force-export=true --force-overwrite=true --report cuda_gpu_kern_sum --format csv --output . "%TARGET%.nsys-rep"
if errorlevel 1 exit /b 1
popd

:end
endlocal
