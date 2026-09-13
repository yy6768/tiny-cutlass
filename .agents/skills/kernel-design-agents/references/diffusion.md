# 面向端侧 Diffusion 的算子优化路线

具体模型、权重、分辨率和部署框架尚未指定。下面是建立任务契约和筛选实验的框架，
所有优先级都要由实际模型 trace 修正；不预填性能和质量结果。

## 先记录模型工作负载

记录模型与 revision、U-Net / DiT 或其他结构、图像/视频分辨率、batch、token 数、
head dimension、prompt 长度、CFG 实现、采样器、步数、timestep 分布、dtype、
框架/编译模式、是否 CUDA Graph、权重是否常驻、显存/功耗/延迟预算。

从一次完整推理和稳定重复推理中采集算子名称、调用次数、输入 shape/stride/layout、
累积耗时与临时张量。至少覆盖真实高频 shape、边界 shape 和内存压力最大的用例。
热度按累计耗时及关键路径判断；同一算子被多步重复调用时，不只看单次最大 kernel。

模型结构可对照原始资料：[Latent Diffusion](https://arxiv.org/abs/2112.10752)、
[DiT](https://arxiv.org/abs/2212.09748)。具体实现仍以用户模型源码为准。

## 候选池

| 家族 | 需要采集的 shape / 语义 | 起步实验 | 可复用的仓库区域 |
|---|---|---|---|
| Attention | B、heads、Lq/Lk、D、scale、mask、layout | 避免显式分数矩阵、调 tile、减少重排 | `csrc/flash-attention/` |
| QKV / output projection / MLP | M/N/K、stride、bias、activation、gate | TensorOp GEMM、epilogue fusion | `csrc/swin/swin_mlp/`、CUTLASS GEMM examples |
| AdaLN / LayerNorm / RMSNorm / GroupNorm | reduction axes、epsilon、affine、调制 broadcast | reduction 与相邻 scale/shift/residual 融合 | `csrc/swin/layernorm/`；新语义另建 family |
| U-Net / VAE convolution | N/H/W/C、R/S、groups、stride、padding、layout | cuDNN 对照、implicit-GEMM、适用的 epilogue | `csrc/conv-fused/` |
| Pointwise / layout / embedding | 张量大小、广播、view 与真实 copy | 消除冗余读写或合并 launch | 按真实算子建立 family |
| Quantization / dequantization | 格式、scale 粒度、packing、rounding、异常值 | 权重预处理、activation 量化与融合 | 明确独立 policy 后接入 |

Diffusion attention 不默认 causal，也不默认采用自回归 decode 的 KV cache。
文本条件与图像 token 的长度可能不同；mask、padding、cross-attention 和 CFG
拼接方式必须来自模型。DiT 不必包含卷积；U-Net/VAE 也不应被简化成纯 GEMM 工作负载。

## 三架构的起步顺序

- **SM89**：优先 FP16/BF16 热点、访存和 launch 代价，考察 fusion 与小 batch 的并行度。
- **SM90**：在相同契约上测 TMA/WGMMA 和 schedule；保留与端侧设备的可比输入和统计方式。
- **SM120**：先复现 FP16/BF16，再调 SM120 数据通路，最后按收益和质量预算评估 FP8/NVFP4。

这是规划顺序。没有 trace 时，只列候选和需要的证据，不给“首要瓶颈”结论。

## 三层正确性

1. **算子层**：固定可信 reference 和容差。覆盖真实 shape、tail、mask、极值、
   NaN/Inf 处理及多 seed。报告 max absolute error、适当的 relative error、MAE/RMSE；
   近零 reference 的相对误差需明确定义。attention/reduction 优先保留可靠累加精度。
2. **Denoiser 单步**：固定 latent、条件、timestep，比较替换前后的预测输出。
   覆盖早/中/晚 timestep；用于定位融合顺序或量化引入的误差。
3. **完整采样**：固定 prompts、seeds、scheduler、steps、CFG、分辨率与软件版本。
   按任务选择图像误差、感知/语义指标和人工样例；FID 等分布指标需要合适的数据量，
   少量图片不能支持分布质量结论。质量阈值在测量前写入契约。

仓库 `cutlass-kernel` 的通常 MAE ≤ 1e-3 只是起点；单一 MAE 不足以证明所有算子或
量化模型可接受。算子容差与模型质量阈值分开设定。改变采样步数、调度器、attention
近似算法或跨 timestep cache 会改变模型/算法契约，应作为单独方案处理。

## 公平计时

| 范围 | 应报告的内容 |
|---|---|
| 单 kernel | CUDA event、warmup、重复次数、median 和波动；分配/编译是否排除 |
| 融合子图 | 输入重排、scale/quantize、workspace、split merge 和输出转换全计入 |
| Denoiser 一步 | 调用次数、GPU 时间及必要的 CPU launch 开销 |
| 完整推理 | 冷启动与稳态分别报告；端到端 wall time、峰值显存，必要时功耗/能耗 |

固定 graph on/off、TF32、dtype、精度和 reference 后端；记录 cuBLAS/cuDNN/PyTorch/
TensorRT 等实际 baseline 选择。端侧 batch=1 是需要考虑的情形，不是未见模型就强制的事实。
权重预打包可单列一次性成本，但 activation 量化不能从稳态时间中删除。

若串行工作中某热点占比为 p，该部分加速 s 倍，理想总加速上限为
`1 / ((1 - p) + p / s)`；存在重叠或关键路径变化时用端到端实测解释。
只提高 Tensor Core 利用率而未改善目标延迟、显存或质量，不构成自动晋升理由。

## 第一轮可交付结果

任务契约、带来源的 shape 清单、可信 baseline、一个范围明确的候选、可复现验证/计时命令、
数值与性能证据，以及采用/修改/淘汰的理由。没有模型输入时，先交付契约和热点采集计划；
不要把随机 GEMM 的吞吐当成端侧 Diffusion 收益。
