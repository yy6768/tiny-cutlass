@echo off
setlocal EnableExtensions

set "ROOT=%~dp0..\..\.."
set "BUILD_DIR=%ROOT%\build\catmull-rom"
set "REPORT_DIR=%ROOT%\build\reports\catmull-rom"
set "CONFIG=Release"
set "PYTHON=python"
set "VSDEVCMD=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"

if /I "%CATMULL_ROM_REGENERATE_SLANG%"=="1" (
  if not defined SLANGC set "SLANGC=C:\VulkanSDK\1.4.309.0\Bin\slangc.exe"
  if not exist "%SLANGC%" (
    echo [catmull-rom] slangc not found: %SLANGC%
    exit /b 1
  )
  echo [catmull-rom] regenerate CUDA
  "%SLANGC%" -target cuda -line-directive-mode none ^
    -o "%ROOT%\csrc\catmull-rom\slang\catmull_rom.cu" ^
    "%ROOT%\csrc\catmull-rom\slang\catmull_rom.slang"
  if errorlevel 1 exit /b 1
  powershell -NoProfile -Command "(Get-Content '%ROOT%\csrc\catmull-rom\slang\catmull_rom.cu') -replace 'extern \"C\" __constant__ GlobalParams_0 SLANG_globalParams;', '__constant__ GlobalParams_0 SLANG_globalParams;' | Set-Content -Encoding ascii '%ROOT%\csrc\catmull-rom\slang\catmull_rom.cu'"
  if errorlevel 1 exit /b 1
)

if exist "%VSDEVCMD%" (
  call "%VSDEVCMD%" -arch=x64 -host_arch=x64
  if errorlevel 1 exit /b 1
)

echo [catmull-rom] build
cmake -S "%ROOT%" -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=%CONFIG% -DCMAKE_CUDA_ARCHITECTURES=89 -DTINY_CUTLASS_BUILD_FLASH_ATTENTION=OFF -DTINY_CUTLASS_BUILD_CONV_FUSED=OFF -DTINY_CUTLASS_BUILD_SWIN=OFF -DTINY_CUTLASS_BUILD_NATTEN=OFF -DTINY_CUTLASS_BUILD_CATMULL_ROM=ON
if errorlevel 1 exit /b 1

cmake --build "%BUILD_DIR%" --target catmull_rom
if errorlevel 1 exit /b 1

echo [catmull-rom] verify
%PYTHON% "%ROOT%\scripts\kernels\catmull-rom\verify.py" --build-dir "%BUILD_DIR%" --config "%CONFIG%" --report-dir "%REPORT_DIR%"
if errorlevel 1 exit /b 1

echo [catmull-rom] bench
%PYTHON% "%ROOT%\scripts\kernels\catmull-rom\bench.py" --report-dir "%REPORT_DIR%"
if errorlevel 1 exit /b 1

endlocal
