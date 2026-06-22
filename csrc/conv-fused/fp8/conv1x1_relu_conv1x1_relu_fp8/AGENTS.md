# conv1x1_relu_conv1x1_relu_fp8 记忆

这个文件只记录会影响后续 agent 修改的事实、坑和状态。

## 当前事实

- 用户提出过 `e5m3`，但仓库 CUTLASS 只有 `e4m3` / `e5m2`，没有标准 `e5m3`。
- 本 family 使用 `cutlass::float_e4m3_t`，不要把 PyTorch dtype 写进 core contract。
- 目标硬件是 SM89；CMake 需要为 FP8 源设置 `CUDA_ARCHITECTURES 89`。
- device launch 必须保留 `ArchTag`、input/filter/output element、accumulator/compute type 模板参数；默认实例才固定到 Sm89/e4m3/float。
- ModelOpt 默认 FP8 PTQ 验证属于外部 reference/smoke，不允许重新引入 pybind/ATen adapter。
- 如果后续环境没有 ModelOpt，外部 smoke 允许优雅 skip；不要把依赖缺失误报成 kernel 失败。
- 2026-06-02 曾尝试直接 `python -m pip install --upgrade nvidia-modelopt`，pip resolver 在当前 Windows/PyTorch 2.7.1 环境下失败；之后用 `nvidia-modelopt==0.43.0 --no-deps` 加手动补依赖才导入成功。
- 这次手动补依赖把 `pydantic` 从 `1.10.12` 升到了 `2.13.4`，会和 `anaconda-cloud-auth` 的 `pydantic<2.0` 约束冲突；这是环境事实，不是 conv-fused 代码要求。
- 当前 ModelOpt 导入会提示 triton plugin DLL 初始化失败、transformers plugin 与本地 `transformers==4.32.1` 不兼容；这些警告不影响当前 conv PTQ smoke。
- stage0 scale 当前是 per-tensor 标量语义，在 CUTLASS device wrapper 中展开为 hidden-channel 等值向量给 smem-staged stage0 epilogue。
- ModelOpt FP8 reference artifacts 中 `*_scale_inv = amax / 448`；喂给 CUTLASS plugin 的 `stage0_scale` 应该是 `input_scale_inv * weight0_scale_inv / stage0_scale_inv`，`output_alpha` 应该是 `stage0_scale_inv * weight1_scale_inv / output_scale_inv`。
- TensorRT plugin 验证时，plugin 输出 kFP8 后必须接 `IDequantizeLayer` 再和 ModelOpt/TensorRT reference 的 half output 对齐；不要直接把 raw FP8 bytes 当成最终 reference。
- 当前 kernel policy 必须从 `DefaultConv1x1ReluConv1x1Relu<ArchTag, ...>` 模板工厂实例化；不要新增 primary name 写死 dtype 或 `Sm89` 的 concrete policy struct。
- `kernel/` 层只产出 `CutlassKernel`；FP8 device 必须直接包 example 13 的 `cutlass::conv::device::B2bImplicitGemmConvolution<Kernel>`，不要恢复本地复刻的 generic wrapper，也不要新增只做 alias 的 `device/default_*` wrapper。
- `device/` 层入口必须是 CUTLASS-style class，包含 `Arguments`、`can_implement`、`initialize`、`run`、`operator()`；内部直接包 example 13 的 `B2bImplicitGemmConvolution<Kernel>`，不要恢复 standalone `run_*` free function 或本地复刻 generic device wrapper。
- 不要为单个 CTA/warp `GemmShape` 新增 traits wrapper；shape 简单时直接留在 kernel factory 的模板参数里。
- 不要为 CUTLASS 已经提供的 arch、layout、TensorRef packing 或 swizzle 写本地 helper。
- 当前 SM89 FP8 smem-staged fused path 最终仍落到 CUTLASS `DefaultB2bConv2dFprop<..., IteratorAlgorithm::kOptimized, true>`，不要退回旧的 RF resident path。

## 已知设计坑

- 旧 RF resident fragment path 在第三个 FP8 case `(1,64,2,3), hidden=32, out=32, bias=True, stage0_scale=1.25, output_scale=0.5` 曾稳定出现单点 outlier：index `(0,16,0,0)`，actual dequant `1.25`，expected dequant `0.00390625`，max diff `1.24609375`。切到 CUTLASS smem-accumulator specialization 后该 case 通过，max diff `0.125`。
- `cutlass::arch::OpMultiplyAddFastAccum` 曾被试过，不能修复 RF resident path 的 outlier；当前 smem-staged policy 使用 `OpMultiplyAdd`。
- CUTLASS `LinearCombinationGenericWithScalingAndAbsMax` 只有配合 `EpilogueWithAbsMax` 才会真正处理 `scale_d/amax` 输出；example 13 的 fused conv 默认 epilogue 不是这个路径。
- 第一版不要谎称已支持 amax 统计或动态 scale 写回。
- FP8 输出对齐参考时必须按 e4m3 round-trip 后的 float32 比较，不要直接对 fp8 tensor 调 `torch.testing.assert_close`。
- 如果 FP8 policy 编译失败，保留分层文件和明确的 unsupported 结果，不允许退回 raw CUDA kernel。

## 后续推进

- 若要完整对齐 ModelOpt PTQ 的 amax/scale 产物，需要明确 ModelOpt 导出的 scale 张量布局，并决定是 per-tensor 还是 per-channel。
- 若要动态 amax，需要扩展 final epilogue 或引入 CUTLASS 自带支持 absmax 的 epilogue threadblock。
