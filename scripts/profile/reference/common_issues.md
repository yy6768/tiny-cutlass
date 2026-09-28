# 常见问题 & 排查

收集 profiling 中反复遇到的问题和解决方案。

---

## NCU 权限

### Windows

Windows 上通常**不会**遇到权限问题 — 驱动默认允许用户访问 GPU 性能计数器。

如果遇到 `ERR_NVGPUCTRPERM`（极少见）：
- 以管理员身份运行 CMD / PowerShell
- 确认没有其他 ncu 实例在占用 GPU

### Linux

```bash
# 方法 A: 用 sudo
sudo ncu [...]

# 方法 B: 永久配置
sudo sh -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" > /etc/modprobe.d/ncu.conf'
sudo update-initramfs -u
# 重启后普通用户可用
```

---

## `-k "regex:..."` 没匹配到任何 kernel

1. **用 demangled name。** 模板 kernel 名很长，如 `void my_kernel<(int)8, (int)256>(...)`。用 cuobjdump 确认：
   ```bat
   cuobjdump --dump-function-names your_exe.exe
   ```

2. **转义 regex 元字符。** `<>` 在大多数 regex 引擎中不需要转义，但括号要注意。

3. **Kernel 可能没被 launch。** 不带 ncu 单独运行一次确认 kernel 实际执行了。

4. **用 `--set basic` 快速扫描：**
   ```bat
   ncu --set basic your_exe.exe [args]
   ```
   这会列出所有 kernel launch 和名字。

---

## Source view 空白 / `source_info(pc)` 返回 None

编译时没加 `-lineinfo`（本项目用 `--generate-line-info`）。

验证方法：
```bat
cuobjdump --dump-line-info your_exe.exe | head -20
```

如果输出为空 = 没有 line info。本项目 CMake 中 flash-attention 和 swin target 已默认加 `--generate-line-info`。如果你新加 target 忘了加，在 CMakeLists.txt 中：
```cmake
target_compile_options(your_target PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>)
```

---

## PM Sampling 没数据

1. **没请求该 section。** 确认 ncu 命令包含 `--section PmSampling`。`run_ncu.py` 默认已加。

2. **vGPU / 虚拟化环境。** PM sampling 在虚拟化下不支持。

3. **Kernel 太短（< 20 µs）。** PM 采样间隔约 2µs，短 kernel 只有几个样本，噪声大。

4. **特定 metric 在你的 ncu 版本/驱动/GPU 组合下不可用。** 常见：`pmsampling:sm__throughput.*` 或 `pmsampling:dram__throughput.*` 可能返回空 instance 数组，但 `pmsampling:smsp__warps_issue_stalled_*` 系列通常可靠。`plot_timeline.py` 遇到空数据会打印 "no data"。

---

## NCU 收集很慢

1. **`--set full` 需要 45+ replay passes** — 这是正常的。每个 pass 重放 kernel 一次以收集不同 metric 组。如果 kernel 本身 3ms，完整 profile ~15s；如果 kernel 300ms，那就要等很久。

2. **减少 replay：** 先用 `--set basic`（3-5 passes）看看是否有明显问题，再决定是否需要 full。

3. **不要用 `-G`（debug 编译）。** 它让 kernel 慢 100×，对 perf profiling 无用。

---

## 报告文件 0 KB

- Kernel 没 launch（程序提前退出或条件不满足）
- `-k` regex 没匹配到任何 kernel name
- 程序在 kernel launch 前 crash

排查：先不带 ncu 跑一遍，确认 kernel 正常执行。

---

## Metric 返回 `None`

| 可能原因 | 解决 |
|---|---|
| Metric 名在当前 GPU 不存在 | 用 `action.metric_names()` 搜索实际可用名 |
| 该 metric 所属的 section 没被收集 | 重跑 ncu 加对应 `--section` |
| 硬件不支持（如 SM89 上无 TMEM 相关 metric） | 正常，忽略 |

建议用 `ncu_utils.safe()` 或 `metric_or_none()` 做 fallback。

---

## `ncu_report` import 失败

```bat
rem 查找 ncu_report 模块位置
dir /s "C:\Program Files\NVIDIA Corporation\Nsight Compute*\extras\python\ncu_report.py"

rem 设置 PYTHONPATH
set PYTHONPATH=%PYTHONPATH%;C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.2.0\extras\python
python -c "import ncu_report; print('OK')"
```

`ncu_utils.py` 会自动尝试常见路径。如果仍失败，检查：
- `ncu_report.py` 旁的 `_ncu_report*.pyd`（Windows）或 `.so`（Linux）是否匹配你的 Python 版本
- 尝试用 Nsight Compute 自带的 Python（通常在同目录下）

---

## 结果在多次运行间抖动

1. **锁定 GPU 频率：**
   ```bat
   rem 查看当前频率
   nvidia-smi -q -d CLOCK

   rem 锁定到 boost 频率（需要管理员）
   nvidia-smi -lgc <boost_clock_mhz>

   rem 解锁
   nvidia-smi -rgc
   ```

2. **关闭后台 GPU 任务**（游戏、浏览器硬件加速等）。RTX 4070/4090 Laptop 尤其容易受影响。

3. **启用 persistent mode：**
   ```bat
   nvidia-smi -pm 1
   ```

4. **ncu 版本不一致。** 不同 ncu 版本可能用不同 metric 名或计算方式。确认团队用同一版本。

---

## "sm__throughput = X%"，算好吗？

取决于 kernel 类型：
- **GEMM / matmul / attention：** > 50% 良好，< 30% 有优化空间
- **Element-wise / reduction：** 通常 10-30%（DRAM-BW-bound 是正常的）
- **CUTLASS GEMM on SM89：** 参考 cuBLAS 相同 shape 作为 baseline

总是同时看 `dram__bytes_read.sum.pct_of_peak_sustained_elapsed`：
- DRAM 饱和 + SM 低 → 正常，bandwidth-bound
- DRAM 低 + SM 低 → latency-bound，有优化空间

---

## "Est. Speedup: X%" 靠谱吗？

基本靠谱。NCU 规则引擎的估计通常在正确的量级。注意：
- 多条规则的 Est. Speedup **不能直接相加**（互相有 overlap）
- 规则不知道修复的工程难度 — 它只估算 perf 影响
- 按 Est. Speedup 排序 = 按影响大小排序，这是正确的优先级

---

## CUTLASS kernel 特有问题

### Kernel 名字太长，regex 难匹配

CUTLASS kernel 的 demangled name 包含大量模板参数，可能有几百个字符。建议：
```bat
rem 先查看实际名字
cuobjdump --dump-function-names build\...\Release\your_target.exe | findstr "cutlass"

rem 用关键字片段匹配
ncu -k "regex:.*Gemm.*Sm89.*" -c 1 ...
```

### 多个 kernel 在同一个 exe 中

用 `--launch-skip-before-match` 和 `--launch-count` 精确定位：
```bat
rem 跳过前 2 次匹配的 launch，只 profile 第 3 次
ncu -k "regex:.*my_gemm.*" --launch-skip-before-match 2 --launch-count 1 ...
```

### 编译时间长

CUTLASS template-heavy 代码编译慢是常态。如果只想 profile 一个特定实例化：
1. 用 `harness_template.cu` 做独立 harness
2. 只 include 需要的头文件
3. 只做一个显式实例化
4. 编译速度从分钟级降到秒级
