# Helpers — 可复用代码

辅助 profiling harness 构建和报告分析。

## C++ / CUDA

| 文件 | 用途 |
|---|---|
| `harness_template.cu` | 独立 profiling harness 起始模板。复制到 `profile/<run_name>/harness/` 下，填写 `TODO(you)` 部分 |
| `safetensors_loader.h` | Header-only safetensors 读取器（零外部依赖）。用于从 `.safetensors` 文件加载真实 workload tensor |

### 典型用法

```bat
cd profile\<run_name>\harness
copy scripts\profile\helpers\harness_template.cu my_kernel_harness.cu
copy scripts\profile\helpers\safetensors_loader.h .
rem 编辑 my_kernel_harness.cu：include 你的 kernel + 填写 main()
nvcc -O2 -std=c++17 --generate-line-info ^
     -gencode=arch=compute_89,code=sm_89 ^
     my_kernel_harness.cu -o my_kernel_harness.exe
```

### 什么时候用 harness（而不是直接 profile CMake target）

- Kernel 在 JIT 编译系统中（TVM、PyTorch inline），无法加 `-lineinfo`
- CMake 整体重编太慢，只想快速迭代单个 kernel
- 想精确控制输入 shape 和数据（如从 safetensors 加载特定 workload）
- 想隔离单个 kernel 的 profile（排除其他 kernel 的干扰）

大部分情况下，本项目的 CMake target 已带 `--generate-line-info`，
可以直接用 `run_ncu.py` profile exe。Harness 是备选方案。

## Python

| 文件 | 位置 | 用途 |
|---|---|---|
| `ncu_utils.py` | `scripts/profile/` | 核心工具：加载报告、安全读取 metric、SM89 key metrics 列表 |
| `analyze_report.py` | `scripts/profile/` | 提取 key metrics + 多报告对比 |
| `extract_stalls.py` | `scripts/profile/` | Per-line stall hotspots（需要 source-level 报告） |
| `plot_timeline.py` | `scripts/profile/` | ASCII PM sampling 时序图（揭示 tail effect、pipeline bubble） |
| `run_ncu.py` | `scripts/profile/` | 统一入口：创建目录 → ncu 收集 → 导出 → 分析 |

### 典型 Python 工作流

```bat
rem 一步到位（推荐）
python scripts/profile/run_ncu.py ^
    --run-name my_kernel_baseline ^
    --exe build\csrc\...\Release\my_target.exe ^
    --kernel-regex "my_kernel.*" ^
    --args "--batch_size=1 --seq_length=1024" ^
    --set both --count 1

rem 手动分步（已有 .ncu-rep 时重新分析）
python scripts/profile/run_ncu.py ^
    --run-name my_kernel_baseline ^
    --exe dummy ^
    --analyze-only
```

所有脚本的输出都在 `profile/<run_name>/analysis/` 下。
