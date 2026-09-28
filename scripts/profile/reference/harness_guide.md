# Harness 构建指南

**Profiling harness** = 一个小型独立 CUDA 可执行文件，唯一目的是用真实输入启动你要 profile 的 kernel，并带有 `-lineinfo` 编译标记。

---

## 什么时候需要 harness

| 场景 | 需要 harness？ |
|---|---|
| 本项目的 CMake target（flash-attention、swin 等） | **不需要** — 已有 `--generate-line-info`，直接用 `run_ncu.py` |
| CUTLASS CMake 重编太慢，想快速迭代单个 kernel | **推荐** — harness 编译 < 5 秒 |
| TVM-FFI / PyTorch inline 编译的 kernel | **必须** — 无法注入 `-lineinfo` |
| 想隔离单个 kernel 避免其他 kernel 干扰 | **推荐** |
| 想用 safetensors 加载真实 workload tensor | **推荐** |

---

## Harness 模板

复制 `scripts/profile/helpers/harness_template.cu` 到你的 run 目录：

```bat
set RUN_DIR=profile\my_kernel_v1
mkdir %RUN_DIR%\harness
copy scripts\profile\helpers\harness_template.cu %RUN_DIR%\harness\my_kernel_harness.cu
copy scripts\profile\helpers\safetensors_loader.h %RUN_DIR%\harness\
```

然后编辑 `my_kernel_harness.cu`，填写 4 个 `TODO(you)` 部分：
1. Kernel 源码（粘贴或 `#include`）
2. 显式模板实例化
3. 输入分配 & 填充
4. Launch 函数

---

## 编译

```bat
nvcc -O2 -std=c++17 --generate-line-info ^
     -gencode=arch=compute_89,code=sm_89 ^
     %RUN_DIR%\harness\my_kernel_harness.cu ^
     -o %RUN_DIR%\harness\my_kernel_harness.exe
```

关键 flag：
- `--generate-line-info`：**必须**，否则 source-level 分析无数据
- `-gencode=arch=compute_89,code=sm_89`：目标 SM89
- `-O2`：保持优化，不要用 `-G`（debug 会让 kernel 慢 100×）

---

## 输入数据策略

### Level 1: 未初始化（最快但最不可靠）

`cudaMalloc` 后不填数据。仅当 kernel 无 data-dependent branch 时可用。

### Level 2: 随机合成（默认推荐）

```cpp
fill_half_random(h_input, 0xA0A0ULL, 0.5f);  // [-0.5, 0.5] 范围
```

Shape 必须匹配实际 workload 的代表性 shape。

### Level 3: 真实数据（safetensors）

```cpp
#include "safetensors_loader.h"

SafetensorsFile st = SafetensorsFile::load("path/to/workload.safetensors");
std::memcpy(h_input.data(), st.tensor_bytes("input_name"), nbytes);
int M = (int)st.entry("input_name").shape[0];
```

用于 kernel 有 data-dependent branches 时，或需要确保数值精度对比。

---

## 显式模板实例化

CUTLASS kernel 通常高度模板化。没有显式实例化的变体会被链接器 strip，ncu 的 `-k regex` 找不到：

```cpp
// 必须为每个要 profile 的 template 参数组合写显式实例化
template __global__ void my_gemm_kernel<
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::half_t
>(const cutlass::half_t*, const cutlass::half_t*, cutlass::half_t*, int, int, int);
```

---

## Profile 前的 sanity check

```bat
rem 先不带 ncu 跑一次，确认 harness 正常工作
%RUN_DIR%\harness\my_kernel_harness.exe 128 128 64
rem 期望输出：
rem [harness] launching kernel...
rem [harness] done.
```

如果 crash 或 hang，先修复再加 ncu — ncu 的错误信息远不如 CUDA runtime 清晰。

---

## 然后 profile

```bat
python scripts/profile/run_ncu.py ^
    --run-name my_kernel_v1 ^
    --exe %RUN_DIR%\harness\my_kernel_harness.exe ^
    --kernel-regex "my_gemm_kernel.*" ^
    --args "128 128 64" ^
    --set both --count 1
```

或者用 harness 的 `--workload` 模式加载 safetensors：
```bat
python scripts/profile/run_ncu.py ^
    --run-name my_kernel_v1_real_data ^
    --exe %RUN_DIR%\harness\my_kernel_harness.exe ^
    --kernel-regex "my_gemm_kernel.*" ^
    --args "--workload path\to\data.safetensors" ^
    --set both --count 1
```
