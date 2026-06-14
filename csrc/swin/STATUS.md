# Swin Status

## 2026-06-12

- Swin 主实现已经重排为 CUTLASS-style 分层：
  `device::Swin<kernel::DefaultSwin<ArchTag, Element, ...>::Kernel>`。
- 公共接口不再提供把 dtype、arch 或 layout 写进名字的 concrete alias。
- 当前测试、plugin 和显式实例化选择：
  `device::Swin<kernel::DefaultSwin<cutlass::arch::Sm80, cutlass::half_t>::Kernel>`。
- 这个选择只出现在调用点、显式实例化和验证路径；不是新的公共类型名，也不是
  fallback alias。
- `threadblock/layout.h` 承担 Swin layout glue：window partition/reverse、
  patch merge、QKV bias split、output bias、PatchEmbed channel/filter pad 和
  BiasLayerNorm。
- `warp/layout.h` 只承担坐标映射，不混入 dtype、arch 或 launcher 逻辑。
- `kernel/default_swin.h` 保存模板 policy factory；`kernel/swin.h` 只保留
  threadblock stage 的 CUDA launch wrapper。
- `swin.cu` 负责把 device facade 映射到 CUTLASS GEMM、fused
  WindowAttention 和 threadblock glue stage。

## ONNX / TensorRT

- `csrc/swin/trt/swin_plugin.{h,cpp}` 提供 TensorRT V2DynamicExt plugin
  `tiny_cutlass_swin`，封装当前 CUTLASS Swin WindowAttention 主路径。
- 当前 plugin activation input/output 只接受 TensorRT `kHALF` + `kLINEAR`，
  并在 `enqueue` 里检查 shape 为 `[B,H,W,C]`；权重、bias 和 workspace 都是
  `cutlass::half_t`。
- 正式 Swin workflow 只保留 `scripts/kernels/swin/{verify.py,bench.py,swin.bat}`；
  旧的 `scripts/swin/` ONNX/TRT 实验脚本已经移除，避免在提交前留下游离入口。
- 当前本机 Python 环境缺少 `tensorrt` 和 `pycuda` 包，所以这条 Python
  TensorRT parity 不能在本次复跑；源码和 C++ plugin build 仍然作为 NHWC/FP16
  包装约束的当前证据。
- 整网 logits parity 还没有作为最终 gate 通过；继续以 reference parity 为先，
  不记录性能结论。

## 最近验证和性能

- `cmake --build build --config Release --target swin` 通过，生成
  `swin_window.exe`、`swin_patch_embed.exe` 和 `swin_plugin.dll`。
- `python scripts\kernels\swin\verify.py --build-dir build --config Release` 通过：
  PatchEmbed host reference `MAE=0.000141383`；stage2 shifted attention cuDNN
  reference `MAE=0`；tail stage cuDNN reference `MAE=1.03903e-09`。
- Tail stage benchmark 修复：`csrc/tests/swin/window.cpp` 现在只在偶数
  feature map 上启用 optional `patch_merged`。Swin 最后一层
  `{B,I,window,shift,heads,head_dim}={1,7,7,0,24,32}` 不再误触发
  patch merge。
- `python scripts\kernels\swin\bench.py --build-dir build --config Release --batch-size 1 --iterations 100 --patch-iterations 50 --nsys --nsys-iterations 3`
  通过，并写入 `build\reports\swin\bench_b1.{csv,json}`。
- 最新 RTX 4070 Laptop GPU / SM89 数字：
  PatchEmbed `0.0896493 ms` / `322.383 GFLOPs`；
  stage0 shift0 `0.115456 ms` / `2525.65 GFLOPs`；
  stage0 shift3 `0.123136 ms` / `2368.12 GFLOPs`；
  stage1 shift0 `0.0708899 ms` / `3687.49 GFLOPs`；
  stage1 shift3 `0.0751206 ms` / `3479.82 GFLOPs`；
  stage2 shift0 `0.0718707 ms` / `3427.1 GFLOPs`；
  stage2 shift3 `0.108288 ms` / `2274.57 GFLOPs`；
  stage3 tail `0.0540774 ms` / `4415.15 GFLOPs`。
- Nsight Systems artifacts: `build\reports\swin\*.nsys-rep`，并导出了
  `cuda_gpu_kern_sum` 和 `cuda_api_sum` CSV。

## PatchEmbed CUTLASS Path

- 新增 `device::PatchEmbed<kernel::DefaultPatchEmbed<ArchTag, Element, ...>::Kernel>`。
- PatchEmbed 路径为：
  `NHWC texture input -> channel pad -> CUTLASS implicit GEMM Conv2d -> BiasLayerNorm -> NHWC output`。
- RGB `C=3` 通过显式 pad 到 `input_channels_padded=8` 进入 TensorOp optimized
  NHWC conv；filter 的 padding 通道补零，不使用 SIMT 或 cuDNN fallback。
- 验证目标：
  `build\csrc\swin\Release\swin_patch_embed.exe --batch_size=1 --image_size=224 --in_channels=3 --input_channels_padded=8 --embed_dim=96 --patch_size=4 --iterations=50`
  通过，host reference `MAE=0.000141383`，`max_abs=0.00390625`。
