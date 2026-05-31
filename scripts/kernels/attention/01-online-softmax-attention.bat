@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0..\..\.."
set "BUILD_DIR=%ROOT%\build"
set "CONFIG=Release"
set "TARGET=flash_attention_01_online_softmax_attention_test"
set "EXE=%BUILD_DIR%\csrc\flash-attention\01-online-softmax\%CONFIG%\%TARGET%.exe"

cmake -S "%ROOT%" -B "%BUILD_DIR%" -DTINY_CUTLASS_BUILD_FLASH_ATTENTION=ON -DTINY_CUTLASS_BUILD_CONV_FUSED=OFF
if errorlevel 1 exit /b 1

cmake --build "%BUILD_DIR%" --config "%CONFIG%" --target "%TARGET%"
if errorlevel 1 exit /b 1

"%EXE%" --head_number=1 --batch_size=1 --head_size=32 --head_size_v=32 --seq_length=32 --seq_length_kv=32 --iterations=1 --reference-check=true
if errorlevel 1 exit /b 1

"%EXE%" --head_number=12 --batch_size=16 --head_size=64 --head_size_v=64 --seq_length=1024 --seq_length_kv=1024 --iterations=20 --reference-check=false

if /I "%CUTLASS_PROFILE%"=="1" goto profile
goto end

:profile
set "REPORT_DIR=%BUILD_DIR%\reports\flash-attention\01-online-softmax"
if not exist "!REPORT_DIR!" mkdir "!REPORT_DIR!"

set "NCU_BASE=!REPORT_DIR!\%TARGET%"
ncu --force-overwrite --set full --launch-skip 6 --launch-count 3 --page raw --csv --import-source=1 --import-sass=1 -o "!NCU_BASE!" "%EXE%" --head_number=12 --batch_size=16 --head_size=64 --head_size_v=64 --seq_length=1024 --seq_length_kv=1024 --iterations=20 --reference-check=false > "!NCU_BASE!.csv"
if errorlevel 1 exit /b 1

set "NSYS_BASE=!REPORT_DIR!\%TARGET%"
nsys profile --force-overwrite=true --trace=cuda,nvtx -o "!NSYS_BASE!" "%EXE%" --head_number=12 --batch_size=16 --head_size=64 --head_size_v=64 --seq_length=1024 --seq_length_kv=1024 --iterations=20 --reference-check=false
if errorlevel 1 exit /b 1

pushd "!REPORT_DIR!"
nsys stats --force-export=true --force-overwrite=true --report cuda_gpu_kern_sum --format csv --output . "%TARGET%.nsys-rep"
if errorlevel 1 exit /b 1
nsys stats --force-export=true --force-overwrite=true --report cuda_api_sum --format csv --output . "%TARGET%.nsys-rep"
if errorlevel 1 exit /b 1
popd

:end
endlocal
