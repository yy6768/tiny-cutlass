@echo off
setlocal EnableExtensions

rem EVT (Epilogue Visitor Tree) learning kernels.
rem Required order: build -> verify -> bench. Verification failure stops before bench.

set "ROOT=%~dp0..\..\.."
set "BUILD_DIR=%ROOT%\build\evt"
set "REPORT_DIR=%ROOT%\build\reports\evt"
set "CONFIG=Release"
set "EXE=%BUILD_DIR%\tests\evt\evt_gemm_broadcast.exe"
set "VSDEVCMD=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"

if exist "%VSDEVCMD%" (
  call "%VSDEVCMD%" -arch=x64 -host_arch=x64
  if errorlevel 1 exit /b 1
)

echo [evt] build
cmake -S "%ROOT%" -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=%CONFIG% -DCMAKE_CUDA_ARCHITECTURES=89 -DTINY_CUTLASS_BUILD_FLASH_ATTENTION=OFF -DTINY_CUTLASS_BUILD_CONV_FUSED=OFF -DTINY_CUTLASS_BUILD_SWIN=OFF -DTINY_CUTLASS_BUILD_NATTEN=OFF -DTINY_CUTLASS_BUILD_CATMULL_ROM=OFF -DTINY_CUTLASS_BUILD_EVT=ON
if errorlevel 1 exit /b 1

cmake --build "%BUILD_DIR%" --target evt
if errorlevel 1 exit /b 1

echo [evt] verify
"%EXE%" --verify
if errorlevel 1 exit /b 1

echo [evt] bench
"%EXE%" --m=4096 --n=4096 --k=2048 --iterations=100
if errorlevel 1 exit /b 1

rem Note: compare with the quotes on the LEFT of == too, so that a caller doing
rem `set EVT_PROFILE=1 && ...` (which captures a trailing space) still matches.
if /I "%EVT_PROFILE:  = %"=="1" goto :profile
if /I "%EVT_PROFILE%"=="1" goto :profile
goto :done

:profile
echo [evt] profile
if not exist "%REPORT_DIR%" mkdir "%REPORT_DIR%"
ncu --set full --force-overwrite -o "%REPORT_DIR%\gemm_broadcast" "%EXE%" --m=4096 --n=4096 --k=2048 --iterations=0
if errorlevel 1 exit /b 1
ncu --import "%REPORT_DIR%\gemm_broadcast.ncu-rep" --csv --page details > "%REPORT_DIR%\gemm_broadcast.csv"
if errorlevel 1 exit /b 1

:done

endlocal
