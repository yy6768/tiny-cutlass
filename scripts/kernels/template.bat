@echo off
setlocal

rem Copy this file into scripts\kernels\<family>\<variant>.bat and replace:
rem   TARGET, EXE_SUBDIR, VERIFY_ARGS, BENCH_ARGS.
rem The required order is build -> verify -> bench.
rem Optionally gate Nsight profiling with CUTLASS_PROFILE=1 and write reports
rem under build\reports\...

set "ROOT=%~dp0..\.."
set "BUILD_DIR=%ROOT%\build"
set "CONFIG=Release"
set "TARGET=<cmake-target>"
set "EXE=%BUILD_DIR%\<exe-subdir>\%CONFIG%\%TARGET%.exe"
set "VERIFY_ARGS=<small-reference-check-args>"
set "BENCH_ARGS=<benchmark-args>"

cmake -S "%ROOT%" -B "%BUILD_DIR%"
if errorlevel 1 exit /b 1

cmake --build "%BUILD_DIR%" --config "%CONFIG%" --target "%TARGET%"
if errorlevel 1 exit /b 1

"%EXE%" %VERIFY_ARGS%
if errorlevel 1 exit /b 1

"%EXE%" %BENCH_ARGS%
