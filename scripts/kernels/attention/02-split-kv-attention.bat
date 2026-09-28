@echo off
setlocal EnableExtensions
set "ROOT=%~dp0..\..\.."
set "BUILD_DIR=%ROOT%\build\split-kv-study\cmake"
set "EVIDENCE=%ROOT%\build\split-kv-study\validation"
set "EXE=%BUILD_DIR%\csrc\flash-attention\Release\flash_attention_test.exe"

cmake -S "%ROOT%" -B "%BUILD_DIR%" -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=89 -DTINY_CUTLASS_BUILD_FLASH_ATTENTION=ON -DTINY_CUTLASS_BUILD_CONV_FUSED=OFF -DTINY_CUTLASS_BUILD_SWIN=OFF -DTINY_CUTLASS_BUILD_NATTEN=OFF -DTINY_CUTLASS_BUILD_CATMULL_ROM=OFF -DTINY_CUTLASS_BUILD_EVT=OFF
if errorlevel 1 exit /b 1
cmake --build "%BUILD_DIR%" --config Release --target flash_attention_test split_kv_attention --parallel 4
if errorlevel 1 exit /b 1
python -B "%ROOT%\csrc\tests\flash-attention\verify.py" --exe "%EXE%" --output-dir "%EVIDENCE%"
if errorlevel 1 exit /b 1
python -B "%ROOT%\csrc\tests\flash-attention\bench.py" --exe "%EXE%" --output-dir "%EVIDENCE%"
if errorlevel 1 exit /b 1
endlocal
