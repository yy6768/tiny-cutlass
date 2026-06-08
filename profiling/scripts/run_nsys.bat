@echo off
setlocal EnableExtensions

rem -----------------------------------------------------------------------------
rem Repo paths
rem -----------------------------------------------------------------------------
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "REPO_ROOT=%%~fI"

rem -----------------------------------------------------------------------------
rem 1. Build target
rem -----------------------------------------------------------------------------
set "BUILD_DIR=%REPO_ROOT%\build"
set "CONFIG=Release"
set "TARGET=flash_attention_00_naive_attention_test"
set "TARGET_EXE=%BUILD_DIR%\csrc\flash-attention\00-naive-attention\%CONFIG%\%TARGET%.exe"

if /I "%TINY_CUTLASS_PROFILE_SKIP_BUILD%"=="1" goto verify

cmake -S "%REPO_ROOT%" -B "%BUILD_DIR%" -DTINY_CUTLASS_BUILD_FLASH_ATTENTION=ON -DTINY_CUTLASS_BUILD_CONV_FUSED=OFF
if errorlevel 1 exit /b 1

cmake --build "%BUILD_DIR%" --config "%CONFIG%" --target "%TARGET%"
if errorlevel 1 exit /b 1

:verify
if /I "%TINY_CUTLASS_PROFILE_SKIP_VERIFY%"=="1" goto profile

rem -----------------------------------------------------------------------------
rem 2. Correctness gate
rem -----------------------------------------------------------------------------
set "VERIFY_ARGS=--batch_size=1 --head_number=1 --head_size=32 --head_size_v=32 --seq_length=32 --seq_length_kv=32 --iterations=1 --reference-check=true"
"%TARGET_EXE%" %VERIFY_ARGS%
if errorlevel 1 exit /b 1

:profile
rem -----------------------------------------------------------------------------
rem 3. Executable arguments
rem -----------------------------------------------------------------------------
set "ITERATIONS=1"
set "REFERENCE_CHECK=false"
set "TARGET_ARGS=--batch_size=1 --head_number=32 --head_size=256 --head_size_v=256 --seq_length=1024 --seq_length_kv=1024 --iterations=%ITERATIONS% --reference-check=%REFERENCE_CHECK%"

rem -----------------------------------------------------------------------------
rem 4. Report path
rem -----------------------------------------------------------------------------
set "REPORT_DIR=%BUILD_DIR%\reports\profiling\flash-attention\00-naive-attention\nsys"
set "CSV_DIR=%REPORT_DIR%\csv"
if not exist "%CSV_DIR%" mkdir "%CSV_DIR%"
set "REPORT_PATH=%REPORT_DIR%\nsys_00_naive_attention"
set "REPORT_FILE=%REPORT_PATH%.nsys-rep"

rem -----------------------------------------------------------------------------
rem 5. Profiling trace / filters
rem    Default GPU metric set targets Ada. Override NSYS_GPU_METRICS_SET if needed.
rem -----------------------------------------------------------------------------
if not defined NSYS_GPU_METRICS_SET set "NSYS_GPU_METRICS_SET=ad10x"
set "NSYS_PROFILE_ARGS=profile --force-overwrite=true --trace=cuda,nvtx --gpu-metrics-devices=0 --gpu-metrics-set=%NSYS_GPU_METRICS_SET% --gpu-metrics-frequency=10000"

rem -----------------------------------------------------------------------------
rem 6. Log
rem -----------------------------------------------------------------------------
echo [nsys] target exe:
echo   "%TARGET_EXE%"
echo [nsys] target args:
echo   %TARGET_ARGS%
echo [nsys] report:
echo   "%REPORT_FILE%"
echo [nsys] profiler args:
echo   %NSYS_PROFILE_ARGS%
echo.

nsys %NSYS_PROFILE_ARGS% -o "%REPORT_PATH%" -- "%TARGET_EXE%" %TARGET_ARGS%
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" exit /b %EXIT_CODE%

pushd "%CSV_DIR%"
nsys stats --force-export=true --force-overwrite=true --report cuda_gpu_kern_sum --format csv --output . "%REPORT_FILE%"
if errorlevel 1 exit /b 1
nsys stats --force-export=true --force-overwrite=true --report cuda_api_sum --format csv --output . "%REPORT_FILE%"
if errorlevel 1 exit /b 1
popd

echo.
echo [nsys] exit code: 0
exit /b 0
