# 720p Swin block 优化契约

用户目标：输入 `[1,720,1280,32]`，4×4 window，每窗口 16 个 token，
MLP ratio=4，即 32→128→32。默认 QK shared，G=C/Dv=1，Dq=Dv=32；
同时验证 separate QK、无 shift 和 reflect shift=(2,2)。输入是已有 32-channel
特征，不是 RGB 图像。无 shift 有 57,600 windows；shift 补边后另计 halo。

完整计算为 `r=x+attention(RMSNorm(x))`，
`y=r+fc2(GELU(fc1(RMSNorm(r))))`。两个 norm 的 gamma 独立，
位置偏置提前展开为 [G*16,16]。推理 FP16/FP32 accumulate，保留 eager
linear、GELU、residual 的 FP16 舍入点，GELU 为 erf 版本，dropout=0。

现有 attention 不能证明完整 block 性能。新增 block API 放在 window_attention
family，复用 CUTLASS warp MMA；不调用旧 LayerNorm MLP 实验。
reference 为独立 PyTorch 完整 block，MAE≤1e-3，max_abs≤2e-2，非有限即失败。

候选：00 四 warp/窗口完整融合；01 一 warp/窗口并让固定 C/G/Dq/Dv 成为
编译期形状。该尺寸只需一组 attention，00 的其他 warp 在 attention 阶段空闲。
MLP 阶段与 attention 阶段的 shared storage 通过 union 复用，residual 独立保存。
所有矩阵乘法使用 CUTLASS TensorOp，不加 SIMT GEMM fallback。

计时覆盖完整 block，使用 CUDA Graph replay 去除 Python/host dispatch 差异；
重复交替采集 baseline/candidate，记录 median、min/max，另提供 PyTorch graph
结果。输入/输出各 58,982,400 bytes，需在 GPU 上验证全尺寸而非抽样替代。
实际设备 SM89 RTX 4070 Laptop；不宣称已测 4090。
