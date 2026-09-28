# SM89 padded window attention：现状与审查

2026-09-15。此轮只落地 KDA、规则、源码门禁和后续契约，不修改 kernel、公共 API 或容差。
当前目标是用户实际遇到的 padded window attention，SM89 选择 CUTLASS 2.x。其他维护架构待指定。

## 现状证据

| 项目 | 当前源码事实 | 对后续工作的含义 |
|---|---|---|
| 完整 attention | [DefaultWindowAttention](../../window_attention/kernel/default_window_attention.h) 只允许 half，复用 CUTLASS warp MMA | FP16 是现有语义基线，不能只改一个 Element typedef 就宣称支持 FP8 |
| FP8 请求 | [WindowAttentionProblem::valid](../../window_attention/problem.h) 拒绝 `quant_fp8=true` | 完整 FP8 attention 尚未接入 |
| FP8 primitive | [DefaultGroupedGemm](../../window_attention/kernel/default_grouped_gemm.h) 组装 stock GemmGrouped；同格式 E4M3/E5M2、FP32 输出、公共 alpha | 只证明该 primitive；没有 attention 的逐阶段量化/scale 契约 |
| 底层主线 | 当前 16 个本地源码文件使用 CUTLASS；首次源码预检未发现本地 asm/CuTe/3.x、Torch 或 SIMT 路径 | 不预设当前代码混用了全部技术栈；继续检查实际结构问题 |
| 完整 block | [DefaultSwinBlock](../../window_attention/kernel/default_swin_block.h) 和 [SwinBlockMma](../../window_attention/threadblock/swin_block_mma.h) 复用 attention，并添加 RMSNorm、residual、MLP | attention 重构会影响 block，必须有独立回归 |

现有 README/AGENTS 中的历史验证是 source-reported。本轮重新构建和对拍状态记录在
[plan 的实施记录](plan.md#本轮实施记录)；不把独立 grouped GEMM 或完整 block 的历史成绩当成
FP8 attention 结果。reference 是独立 PyTorch 转写，不消费 kernel 的 gather/scatter 表。

## 结构审查

1. [WindowAttentionMma](../../window_attention/threadblock/window_attention_mma.h) 同时处理 reflect
   gather、RMSNorm、QKV、QK、bias/scale/softmax、PV、projection 与 scatter。
   后续应围绕输入准备、GEMM operand/存储、softmax、输出阶段提取职责，保留整体 CTA 数据流；
   不建立多套公开 operator 或用大量空壳包装同一实现。
2. `WarpStorage` 的 a/b、Q/K/V 和 probability 共享一个 `Element`；`mma_step` 固定 K=32、
   每次迭代增加 16，layout 使用 half 相关的 crosswise 配置。FP8 的指令 K、元素大小和 scale
   生命周期需要按阶段 policy 显式表达；PV 未确认量化前不能随着 QK 一起改为 FP8。
3. normalized/hidden 通过 `reinterpret_cast<Element*>(&shared + 1)` 和运行时偏移定位，
   [device::shared_size](../../window_attention/device/window_attention.h) 在另一层计算动态字节数。
   shared-memory 容量、对齐和生命周期应由同一存储契约描述，避免扩充元素类型后两处失配。
4. block 把 attention storage 与 scratch 放在一个 union 分支，调用 attention 写出中间窗口，
   然后经 barrier 复用空间做 MLP。该复用依赖 attention 的尾随存储约定；调整布局必须同时
   检查静态 attention、标量/向量访存、barrier 和 residual 保留范围。
5. 当前 GEMM 之间把 accumulator 写入 shared，再装载下一步 operand。是否值得保留某些
   register 中间值、调整 warp/window 映射或扩大向量访存，需要 NCU 和真实 workload 证明；
   这只是候选问题，不是已定位的性能瓶颈或收益结论。

## 语义基线

连续 `[B,H,W,C]` 输入，window=4（16 tokens）。reflect 支持重复反射，单像素轴复制；
两轴 shift 独立，无 cyclic shift 或 attention mask。halo gather 可读，halo scatter=-1，
只把裁剪后的真实 token 写回。

必须融合 RMSNorm（必需 gamma，默认 epsilon 为 half finfo epsilon）；QK shared/separate，
Dq/Dv 可不同。位置偏置运行前展开为 `[G*16,16]`，保留 `(QKᵀ+bias)*scale` 顺序。
当前 eager FP16 在 QKV、scores、bias 加法、scale、softmax 和 PV 的舍入点必须保持。
核心参数是调用者所有的 raw device pointers、scalar descriptor 和 stream。

完整 block 的两次 residual、第二个 RMSNorm 和 erf-GELU MLP 属于另一 API/计时范围。
已有 `[1,720,1280,32]` 是完整 block 主用例，不能未经确认就当作 FP8 attention 的唯一 workload。
现有 attention benchmark 是若干小图 eager CUDA event 延迟，也不能代替 720p block 性能。

## FP8 契约仍缺什么

在 [后续计划](plan.md) 中按必填项补齐：量化 reference/真实输入、各阶段 FP8 格式及量化位置、
scale 定义和粒度、舍入/饱和、PV 的 operand 类型、误差/模型质量标准、主性能 workload 与计时口径。
不能把现有 FP16 MAE/max_abs，或对已量化输入验证的 grouped GEMM 容差，直接当作完整 FP8 模型容差。
