# Fused window attention

完整 block 新入口是 `ops/swin_block.h` 的 `swin_block<Element>()`。针对用户指定的
`[1,720,1280,32]`、window=4、MLP ratio=4，单 kernel 完成
`r=x+attention(RMSNorm(x))`，再执行 `r+fc2(GELU(fc1(RMSNorm(r))))`。
两个 norm 使用独立 gamma；GELU 为 erf 版本，linear 和残差保留 FP16 舍入。
本文件后文的 `window_attention()` 仍只计算 norm_attn + attention。

```bat
scripts\kernels\swin\swin.bat block
```

该入口 build → 23 个完整 block 对拍（含三个完整 720p 用例）→ 三个 720p
Graph benchmark。可用 `CUTLASS_PROFILE=1` 附加 NCU/NSYS；产物在
`build/swin/reports/block/`，本次候选对比和报告在 `build/swin/block/`。

`SwinBlockArguments<Element>` 继承 attention 参数，额外要求
`mlp_rms_weight[32]`、`fc1_weight[128,32]`、`fc1_bias[128]`、
`fc2_weight[32,128]`、`fc2_bias[32]`。所有浮点 buffer 要求 16-byte 对齐；
输出不得与任何输入重叠，两个 index 数组使用已有 reflect builder 生成。
该 block 路径显式支持 C=Dq=Dv=32、G=1、ratio=4；其他配置返回 unsupported。
batch/H/W、两轴 shift 及 shared/separate QK 仍由 descriptor 指定。
RMSNorm、索引/权重准备、dtype/dropout 的约定沿用下面的说明。

一个 warp 负责一个完整窗口；TensorOp operand 使用 128-bit 向量搬运，
RMSNorm 每 lane 处理 8 channels、4 lanes 合作一行。attention 与 MLP 的
shared storage 通过 union 复用，中间结果不写入显存。没有额外 partition、
norm、residual 或 MLP kernel。计时不含一次性的 host 索引/位置偏置准备及 H2D。

实现用户提供的 attention 非量化推理前向，并将外层 `norm_attn` 的 RMSNorm 融入同一个 kernel。输入和输出都是
连续的 `[B,H,W,C]`，完整计算只启动一个 CUTLASS kernel：

```text
reflect gather -> RMSNorm across C (mandatory)
  -> grouped QKV projection
  -> softmax((QK^T + preexpanded_position_bias) / sqrt(Dq))
  -> P @ V -> concatenate groups
  -> output projection + output bias
  -> crop scatter
```

四个矩阵乘法均使用 CUTLASS 2.x `DefaultMmaTensorOp`，通过其原生
`IteratorA/B` 执行 `ldmatrix`、`m16n8k16` TensorOp，使用 `IteratorC`
写回共享内存。QKV、scores、probabilities 和 hidden 均留在片上；没有
中间显存 workspace、独立 partition/reverse kernel 或标量 GEMM fallback。

## 支持范围和参考约定

| 参数 | 当前支持 |
|---|---|
| 元素类型 | `cutlass::half_t`；MMA 和 softmax 使用 FP32 累加 |
| 构建架构 | CMake `CMAKE_CUDA_ARCHITECTURES=80` 或 `89`；实测设备为 SM89 |
| window | 4×4，16 tokens |
| C | 8 的倍数，8～1024 |
| groups | 1～64，且 `groups * value_channels <= 512` |
| QK/value channels | 分别为 8 的倍数，8～64，二者可以不同 |
| QK | shared 或 separate；shared 时每个 group 内 K=Q |
| position bias | 运行前预展开为二维 `[G*16,16]`，所有窗口复用 |
| RMSNorm | 必须提供 `[C]` gamma；FP32 平方和与 rsqrt，无均值扣除和 beta |
| shift | 两轴独立，分别为 0～3 |
| reflect | 支持非整除尺寸和小图，补边后裁回原尺寸 |

QKV 权重与原参考完全同序：`[G,Dq+Dv,C]` 或 `[G,2*Dq+Dv,C]`，没有
projection bias；输出权重为 `[C,G*Dv]`，输出 bias 为 `[C]`，也可传空指针。
`C` 不要求等于 `G*Dv`。

当前接口遵循用户追加的 RMSNorm、reflect 和位置偏置预展开要求：

- `reflect_pad_nhwc` 使用可重复的镜像反射；长度为 1 的轴复制该像素。
- 不使用 cyclic 或 shifted-window mask；descriptor 不再暴露这些选项。
- `rms_epsilon` 默认 `0.0009765625`，对应参考 `nn.RMSNorm(C, eps=None)`
  在 FP16 输入下的 epsilon；允许显式正有限值，gamma 是必需指针。
- relative table `[49,G]` 通过 `position_bias.h` 的
  `expand_relative_position_bias<Element>()` 在 host 按 query-minus-key 展开；
  dense 参数直接重排为同一二维格式。权重更新时重算，运行时只读，计时不含预计算。
- `quant_fp8=False`。FP8 quantizer、训练 dropout、backward 不在该实现中；
  `quant_fp8=True` 或非零 dropout 请求返回 `kErrorNotSupported`。

保留原参考的 **先加 bias，再乘 scale** 顺序。以整个模块转为 FP16 的
PyTorch eager 结果为 reference，QKV、QK、bias 加法、scale、
softmax 和 PV 输出保留对应的 FP16 舍入点。输出投影在 FP32 accumulator
上加 bias 后再转换为 FP16。

该 API 包含 `norm_attn + attention`，调用者应传未经 norm 的输入，避免重复归一化。
两次残差、`norm_mlp` 和 MLP 不在此 API 中。仓库原有标准 Swin 的其他 family
是独立实验，不构成用户参考的完整 `DLSS4SwinBlock`。

## 调用

链接 CMake target `swin_window_attention_core`，包含
`swin/window_attention/ops/window_attention.h`。核心接口不依赖 Torch/ATen。

```cpp
namespace wa = tiny_cutlass::swin::window_attention;
wa::WindowAttentionProblem problem;
problem.batch = 1;
problem.height = problem.width = 64;
problem.channels = 96;
problem.groups = 3;
problem.qk_channels = problem.value_channels = 32;
problem.shift_h = problem.shift_w = 2;

// Host preparation, once per geometry:
auto index = wa::build_window_attention_index(problem);
// Upload index.gather/index.scatter to caller-owned device int arrays.

wa::WindowAttentionArguments<cutlass::half_t> args;
args.problem = problem;
args.input = input_device;
args.rms_weight = rms_gamma_device;
args.qkv_weight = qkv_weight_device;
args.position_bias = expanded_position_bias_device; // [G*16,16]
args.output_weight = output_weight_device;
args.output_bias = output_bias_device;
args.gather = gather_device;
args.scatter = scatter_device;
args.output = output_device;
auto status = wa::window_attention(args, stream);
// Check status and the eventual stream synchronization result.
```

构建 host 索引时另需包含 `swin/window_attention/window_index.h`；
准备 relative bias 时包含 `swin/window_attention/position_bias.h`。
所有指针均由调用者持有，存活到 stream 完成；输入、权重和索引只读。
输出 buffer 不得与任何输入重叠。索引必须由匹配 descriptor 的 builder 生成，
runtime 不读取 device 索引内容做 host 检查。reflect 的 gather 与 scatter
不能混用：halo 的 scatter=-1，只允许真实裁剪范围内的 token 写回。

一个 CTA 对应一个 window，4 个 warp 按 group 分工，完成 PV 后经过一次 CTA
barrier，再合作完成输出投影。每个 warp 使用私有共享内存 scratch；hidden
由整个 CTA 共享。QK/V 均不超过 32 时选择 32 维模板 scratch，其余选择 64 维模板。
每 CTA 使用 `34816 + 32*(C+G*Dv)` 或 `47104 + 32*(C+G*Dv)` bytes
动态共享内存（不含 CUDA 保留开销），归一化输入由全部 group 复用。
CUDA Graph 捕获和非默认 stream 均受测试覆盖。

## 构建、验证和测量

```bat
scripts\kernels\swin\swin.bat
```

入口严格执行 build → verify → bench。它构建 `swin`，先运行原有 5 个
family/索引验证程序，再执行 66 个独立 PyTorch attention 用例。任一验证失败
都会中止，benchmark 不会执行。`run.bat` 是同一入口的转发。

单独执行 attention gate 或 benchmark：

```bat
python -B csrc\tests\swin\verify.py --build-dir build\swin
python -B csrc\tests\swin\bench.py --build-dir build\swin
python -B csrc\tests\swin\bench.py --case window_attention_64x64 --ncu --nsys
```

benchmark 对它实际测量的每个 shape 重新生成 reference，并在 C++ timed loop
之前验证同一组输入。判据为 MAE ≤ 1e-3、max_abs ≤ 2e-2，任一非有限值失败。
两端报告 CUDA event 测量的 eager 调用延迟，包含主机提交造成的 GPU 空隙；
不把该数值当成 TensorOp 峰值吞吐或隔离的 kernel duration。NCU/NSYS 提供
独立的 GPU 时间和资源数据。比较对象是本目录的 PyTorch eager reference；benchmark 同样提前展开位置偏置。

所有 fixture、日志和报告放在 `build/swin/`：

- `reports/window_attention/verify.json`：固定 seed=2026 的逐用例误差。
- `reports/window_attention/bench.{json,csv}`：完整前向耗时。
- `reports/window_attention/ncu/`：`.ncu-rep` 和原始指标 `.csv`。
- `reports/window_attention/nsys/`：`.nsys-rep` 和 kernel summary `.csv`。

在 RTX 4070 Laptop、CUDA 12.9、PyTorch 2.7.1+cu126 上完成了完整对拍；
CUDA Graph 每次只捕获到一个 kernel 节点。使用最大维度 memcheck、reflect racecheck 和多 group synccheck 检查读写及同步；
当前版本的实际运行结果见 `build/swin/profile/` 日志。具体误差和测量值以生成的
报告为准，改动 kernel 后须重新运行。

## FP8 grouped GEMM 与 PTX 验证

`kernel/default_grouped_gemm.h` 提供独立的 `DefaultGroupedGemm<ArchTag,
ElementA, ElementB, ...>` 模板工厂，组装 stock CUTLASS 2.x `GemmGrouped`。
它验证在 SM89 上单次 launch 调度多个 GEMM，并使用原生 FP8 TensorOp。
该 primitive 尚未替换上文的完整 FP16 attention。

需要 CUDA 12.4 及以上、编译目标 SM89。支持 E4M3×E4M3 或 E5M2×E5M2，
FP32 累加寄存器和输出；没有 SIMT 或先转 FP16 的路径。当前 vendored
`GemmGrouped` 的 A/B 指针条件表达式无法编译不同的 operand type，所以
本工厂显式拒绝 E4M3×E5M2 混合格式。硬件本身支持混合 FP8 指令。

测试覆盖两种格式、每种 6 个不同 M/N/K，其中包括 attention 的 `16×16×32`、
`16×32×16`、projection 的 `16×64×96` 和有 tile 尾部的形状。
M 可有尾部；本配置要求 N 整除 4、K 整除 16，A/B 基地址至少 16-byte 对齐。
host 将输入量化成 FP8，再传 raw device pointers 数组和 problem/stride 数组。
对拍 reference 在 CPU 使用 FP64 对同一组已量化输入求乘积；判据 MAE≤1e-3、
max_abs≤1e-2。这个门槛不代表相对未量化模型的误差。

当前 scaling 约定是 `A≈sA*A_fp8`、`B≈sB*B_fp8`，使用公共
`alpha=sA*sB` 在 epilogue 还原 scale；尚不支持每个 group 独立的 scale。
测试另覆盖非零 beta 和 source-C。

```bat
scripts\kernels\swin\swin.bat grouped-gemm
```

同样执行 build → verify → bench。验证还会从实际 executable 导出 PTX/SASS，
检查以下指令，并用 `ptxas -arch=sm_89` 重新组装导出的 PTX：

```ptx
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32
```

SASS 中对应 `QMMA.16832.F32.E4M3.E4M3` 与 `QMMA.16832.F32.E5M2.E5M2`。
输出在 `build/swin/reports/grouped_gemm/`：`grouped_gemm.ptx`、`.sass`、
`.cubin`、`instructions.json` 和验证/benchmark 报告。设置 `CUTLASS_PROFILE=1`
会附加 NCU/NSYS 报告。实测设备是 RTX 4070 Laptop（SM89）；4090 同属 SM89，
但其性能须在对应设备另行测量。

接入用户的 FP8 attention 时，QKV、QK 和 output projection 可以选择 FP8
operand；原始参考的 P 和 V 没有额外 FP8 quantizer，因此 PV 应继续遵循
原参考的类型。直接调用 `GemmGrouped` 会引入 stage 间的显存输出；保留
单 kernel attention 则需要把 FP8 warp MMA 和明确的量化/scale 规则接入
现有 CTA 内的数据流，不能把 grouped GEMM 等同于完整 attention fusion。

## 本次源码报告

`build/swin/profile/index.html` 汇总修正语义后的基线、优化版和 FP8 grouped GEMM。
优化版 NCU full/source 报告在 `rms_reflect_optimized/reports/`，每份报告都嵌入源码；
`analysis/` 提供全量指标 CSV、逐实例 CSV、源码热点 CSV、源码快照和离线 HTML。
报告使用安装在用户目录的 `ncu-report-skill` 生成，按实际 GPU 枚举指标。
SM80/SM120 的 skill 适配不代表本 attention 已在这些 GPU 上完成运行验证。
