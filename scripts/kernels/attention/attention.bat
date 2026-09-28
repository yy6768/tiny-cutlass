@echo off
setlocal EnableExtensions
set "ROOT=%~dp0..\..\.."
set "BUILD_DIR=%ROOT%\build\split-q-study\cmake"
set "EVIDENCE=%ROOT%\build\split-q-study\training-validation"
set "EXE=%BUILD_DIR%\csrc\flash-attention\Release\flash_attention_test.exe"
set "BACKWARD_EXE=%BUILD_DIR%\csrc\flash-attention\Release\flash_attention_backward.exe"

cmake -S "%ROOT%" -B "%BUILD_DIR%" -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=89 -DTINY_CUTLASS_BUILD_FLASH_ATTENTION=ON -DTINY_CUTLASS_BUILD_CONV_FUSED=OFF -DTINY_CUTLASS_BUILD_SWIN=OFF -DTINY_CUTLASS_BUILD_NATTEN=OFF -DTINY_CUTLASS_BUILD_CATMULL_ROM=OFF -DTINY_CUTLASS_BUILD_EVT=OFF
if errorlevel 1 exit /b 1
cmake --build "%BUILD_DIR%" --config Release --target flash_attention_test split_q_attention flash_attention_backward --parallel 4
if errorlevel 1 exit /b 1
python -B "%ROOT%\csrc\tests\flash-attention\verify.py" --kernel=03-split-q --exe "%EXE%" --output-dir "%EVIDENCE%"
if errorlevel 1 exit /b 1
python -B "%ROOT%\csrc\tests\flash-attention\verify.py" --phase=backward --kernel=03-split-q --exe "%BACKWARD_EXE%" --output-dir "%EVIDENCE%"
if errorlevel 1 exit /b 1
python -B "%ROOT%\csrc\tests\flash-attention\bench.py" --kernel=03-split-q --exe "%EXE%" --output-dir "%EVIDENCE%"
if errorlevel 1 exit /b 1
python -B "%ROOT%\csrc\tests\flash-attention\bench.py" --phase=backward --kernel=03-split-q --exe "%BACKWARD_EXE%" --output-dir "%EVIDENCE%"
if errorlevel 1 exit /b 1
endlocal
