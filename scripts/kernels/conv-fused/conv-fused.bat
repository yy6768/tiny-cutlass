@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0..\..\.."
set "BUILD_DIR=%ROOT%\build\conv-fused"
set "CONFIG=Release"
set "VSDEVCMD=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"

if exist "%VSDEVCMD%" (
  call "%VSDEVCMD%" -arch=x64 -host_arch=x64
  if errorlevel 1 exit /b 1
)

cmake -S "%ROOT%" -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=%CONFIG% -DTINY_CUTLASS_BUILD_FLASH_ATTENTION=OFF -DTINY_CUTLASS_BUILD_CONV_FUSED=ON
if errorlevel 1 exit /b 1

cmake --build "%BUILD_DIR%" --target conv_fused conv_pool_test conv_fused_relu_fp8_test
if errorlevel 1 exit /b 1

"%BUILD_DIR%\tests\conv-fused\conv_pool_test.exe"
if errorlevel 1 exit /b 1

"%BUILD_DIR%\tests\conv-fused\conv_fused_relu_fp8_test.exe"
if errorlevel 1 exit /b 1

endlocal
