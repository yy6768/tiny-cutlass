@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0..\..\.."
set "BUILD_DIR=%ROOT%\build"
set "CONFIG=Release"
set "TARGET=swin_cutlass_test"
set "EXE=%BUILD_DIR%\csrc\swin\%CONFIG%\%TARGET%.exe"

cmake -S "%ROOT%" -B "%BUILD_DIR%" -DTINY_CUTLASS_BUILD_FLASH_ATTENTION=OFF -DTINY_CUTLASS_BUILD_SWIN=ON -DTINY_CUTLASS_BUILD_CONV_FUSED=OFF
if errorlevel 1 exit /b 1

cmake --build "%BUILD_DIR%" --config "%CONFIG%" --target "%TARGET%"
if errorlevel 1 exit /b 1

"%EXE%" --batch_size=1 --image_size=14 --window_size=7 --shift_size=3 --head_number=3 --head_size=32 --iterations=1 --reference-check=true
if errorlevel 1 exit /b 1

"%EXE%" --batch_size=8 --image_size=56 --window_size=7 --shift_size=3 --head_number=3 --head_size=32 --iterations=20 --reference-check=false
if errorlevel 1 exit /b 1

if /I "%CUTLASS_PROFILE%"=="1" goto profile
goto end

:profile
set "REPORT_DIR=%BUILD_DIR%\reports\swin"
if not exist "!REPORT_DIR!" mkdir "!REPORT_DIR!"

set "NCU_BASE=!REPORT_DIR!\%TARGET%"
ncu --force-overwrite --set full --launch-skip 6 --launch-count 4 --page raw --csv --import-source=1 --import-sass=1 -o "!NCU_BASE!" "%EXE%" --batch_size=8 --image_size=56 --window_size=7 --shift_size=3 --head_number=3 --head_size=32 --iterations=20 --reference-check=false > "!NCU_BASE!.csv"
if errorlevel 1 exit /b 1

set "NSYS_BASE=!REPORT_DIR!\%TARGET%"
nsys profile --force-overwrite=true --trace=cuda,nvtx -o "!NSYS_BASE!" "%EXE%" --batch_size=8 --image_size=56 --window_size=7 --shift_size=3 --head_number=3 --head_size=32 --iterations=20 --reference-check=false
if errorlevel 1 exit /b 1

:end
endlocal
