# SM89 / SM90 / SM120 架构总览

核对日期：2026-09-09。本文是技能使用入口和迁移规划，不是三架构运行认证。

## 当前仓库与设备

| 项目 | 安装时观察 | 含义 |
|---|---|---|
| Windows 本机 GPU | `Win32_VideoController` 返回 RTX 4070 Laptop GPU | 当前用户目标为 SM89；开始实验时再用 CUDA runtime 查询设备和资源 |
| Toolkit | `nvcc --version`：12.9，V12.9.41 | 编译器存在；不代表每个库、host compiler、驱动组合已验证 |
| CUTLASS | 4.5.2，commit `1732ed7da3b81d9f28b0130370ba70755a7e6dda` | 源码包含 Ada、Hopper、Blackwell GeForce 示例 |
| 根 CMake | 默认 89，allowlist 只有 80、89 | 直接传 90 / 90a / 120 会被拒绝；本次安装不修改 kernel 构建支持 |
| 三架构 parity / 性能 | 本次未运行 CUDA 验证或 benchmark | 后续按 family 和 shape 填证据，不能宣称支持完成 |

本地依据：[根 CMake](../../../../CMakeLists.txt)、
[CUTLASS version](../../../../3rdparty/cutlass/include/cutlass/version.h)、
[CUTLASS 构建条件](../../../../3rdparty/cutlass/CMakeLists.txt)。实际工作时重新读取，不能永久依赖此快照。

## 三条路线

| 维度 | SM89 / Ada | SM90 / Hopper | SM120 / Blackwell RTX |
|---|---|---|---|
| 在本项目中的定位 | 当前端侧实验设备 | 所需的 Hopper 优化与对照平台 | 未来端侧迁移目标 |
| 首轮数值基线 | FP16/BF16，必要处 FP32 累加 | 相同算子契约与输入 | 先复现相同基线，再研究低精度 |
| MMA 主要研究路径 | warp `mma.sync` | `mma.sync` 基线；WGMMA 专用路径 | warp `mma.sync`，含 block-scaled 新指令 |
| 异步数据搬运 | `cp.async` → shared memory | TMA + barrier + warp specialization | 目标库支持的 TMA 路径；不支持 TMA load multicast |
| 工作分工 | CTA/warp tile、stage、epilogue | producer/consumer、异步 MMA、cluster | 单 CTA cluster 起步，按 SM120 schedule 调整 |
| 低精度候选 | FP8，先核对 dtype/layout 和缩放成本 | FP8，逐例核对 builder / scaling | FP8、MX 格式、NVFP4；格式和缩放不能混用 |
| 明确不能搬用 | WGMMA、TMA、TMEM 路线 | SM100 `tcgen05` / TMEM 路线 | SM100 `tcgen05` / TMEM / 2-SM、Hopper WGMMA 路线 |

架构事实来自 [Ada 调优指南](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html)、
[Hopper 调优指南](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)、
[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)，以及仓库固定版本的
[SM120 示例](../../../../3rdparty/cutlass/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu)。
“首轮”“起步”和研究顺序是本项目的工程选择，不是 NVIDIA 的性能保证。

“端侧”在这里指用户要求的 NVIDIA CUDA 设备路线。不要据此给手机 NPU、Jetson 或其他
Blackwell 产品分配 SM120；具体产品必须单独查询 compute capability。
Hopper 的实验结论也必须在端侧目标上重新测量。

## 编译目标不是 GPU 的别名

| 架构 | 常规目标 | 特性目标及约束 |
|---|---|---|
| SM89 | `compute_89` → `sm_89` | 无需添加 `a` 后缀 |
| SM90 | `compute_90` → `sm_90` | WGMMA 路径按 CUTLASS 要求使用 `compute_90a` → `sm_90a` |
| SM120 | `compute_120` → `sm_120` | 本仓库 example 79 接受 `120a` / `120f` 等；不能用普通 `120` 替代其特性目标 |

本地 CUTLASS 的工具链条件是：89/90 从 CUDA 11.8 开始，90a 从 12.0，120/120a 从
12.8，120f 从 12.9。这些是该版本的构建条件，不保证所有示例在这些最低版本都能工作。
`CUTLASS_NVCC_ARCHS` 是第三方 CUTLASS 的选项；根项目使用 `CMAKE_CUDA_ARCHITECTURES`，
两者不能互换。新增目标前检查 CMake 版本和实际生成的 nvcc 命令。

普通 PTX 的前向兼容不能推导出 `compute_90a` 等架构特性 PTX 的前向兼容。
`a` 是架构特性目标；`f` 有 family 范围，也不表示能跨所有 Blackwell 产品运行。
依据：[兼容性说明](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html)、
[PTX target 指令](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#directive-target)、
[example 79 的 CMake 条件](../../../../3rdparty/cutlass/examples/79_blackwell_geforce_gemm/CMakeLists.txt)。

## 开始迁移前的检查

在目标机器记录 CUDA runtime 的 `major/minor`、`multiProcessorCount`、
`totalGlobalMem`、`sharedMemPerBlockOptin`、`sharedMemPerMultiprocessor`、register 限制、
驱动/Toolkit/NCU/CMake/host compiler 和依赖版本。具体 SKU 的 SM 数、显存、带宽、功耗
不从架构名字推断；笔记本记录供电、温度和频率。

随后按 [SM89](sm89.md)、[SM90](sm90.md) 或 [SM120](sm120.md) 选择 policy。
实现时逐 family 检查 arch/dtype/layout/shape，不能只扩展全局 allowlist。
用 target binary / PTX / SASS 和设备运行记录确认实际走到了期望路径。
