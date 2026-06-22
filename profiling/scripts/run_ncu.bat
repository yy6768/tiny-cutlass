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
set "TARGET=naive_attention"
set "TARGET_EXE=%BUILD_DIR%\csrc\flash-attention\%CONFIG%\%TARGET%.exe"

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
set "REPORT_DIR=%BUILD_DIR%\reports\profiling\flash-attention\00-naive-attention\ncu"
set "CSV_DIR=%REPORT_DIR%\csv"
if not exist "%CSV_DIR%" mkdir "%CSV_DIR%"
set "REPORT_PATH=%REPORT_DIR%\ncu_00_naive_attention"
set "REPORT_FILE=%REPORT_PATH%.ncu-rep"
set "CSV_FILE=%CSV_DIR%\ncu_00_naive_attention.csv"

rem -----------------------------------------------------------------------------
rem 5. Profiling metrics / filters
rem    Skip: 3 init kernels + 3 first attention kernels + 3 warmup kernels.
rem    The 3 timed attention kernels are profiled when iterations=1.
rem -----------------------------------------------------------------------------
set "LAUNCH_SKIP_BEFORE_MATCH=9"
set "LAUNCH_COUNT=3"
set "NCU_PROFILE_ARGS=--force-overwrite --target-processes all --set detailed -k regex:attention_.* --launch-skip-before-match %LAUNCH_SKIP_BEFORE_MATCH% --launch-count %LAUNCH_COUNT% --page raw --csv --import-source=1 --import-sass=1"

rem -----------------------------------------------------------------------------
rem 6. Log
rem -----------------------------------------------------------------------------
echo [ncu] target exe:
echo   "%TARGET_EXE%"
echo [ncu] target args:
echo   %TARGET_ARGS%
echo [ncu] report:
echo   "%REPORT_FILE%"
echo [ncu] csv:
echo   "%CSV_FILE%"
echo [ncu] profiler args:
echo   %NCU_PROFILE_ARGS%
echo.

ncu %NCU_PROFILE_ARGS% -o "%REPORT_PATH%" -- "%TARGET_EXE%" %TARGET_ARGS% > "%CSV_FILE%"
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo [ncu] exit code: %EXIT_CODE%
exit /b %EXIT_CODE%
