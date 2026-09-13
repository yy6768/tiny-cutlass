# SM89 / SM90 / SM120 的 Profiling

采用 KDA 的 evidence → diagnosis → plan 思路，替换上游 NCU 技能的 B200 默认值。
reference parity 是前置条件；NCU 数据解释性能，普通运行结果用于判定收益。

## 先确定采集对象

记录目标 GPU、NCU 版本、实际 dispatch、输入、warmup 后的目标 launch 和需要回答的问题。
优化编译带 `-lineinfo`，harness 在 `csrc/tests/`，二进制在 `build/`。
多 kernel 融合子图既测总耗时，也分别定位热点；不要把第一个匹配的初始化 kernel 当目标。

先查本机能力：

```powershell
ncu --version
ncu --list-sets
ncu --list-sections
ncu --query-metrics
```

依据：[NCU CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)。
在 Windows 找不到 `ncu` 时查实际 Nsight Compute 安装目录；不假定 Linux 路径或 B200
NCU 版本。只使用本机可用 section/metric，不通过复制别的 GPU 指标名补全缺项。

## 命令模板

下面变量必须先填入本次 run 的绝对路径、真实 kernel regex 和 harness 参数。
`$RunDir` 应是新建的 `build/experiments/<family>/<run-id>`；benchmark 和 parity 已通过。

```powershell
# 已赋值：$RunDir、$Binary、$KernelRegex、$HarnessArgs（参数数组）。
$ReportDir = Join-Path $RunDir 'profile'
$AnalysisDir = Join-Path $RunDir 'analysis'
New-Item -ItemType Directory -Path $ReportDir,$AnalysisDir -Force | Out-Null
$Prefix = Join-Path $ReportDir 'candidate-overview'
ncu --set basic -k "regex:$KernelRegex" -c 1 -o $Prefix $Binary @HarnessArgs
if ($LASTEXITCODE -ne 0) { throw 'NCU collection failed' }
ncu --import "$Prefix.ncu-rep" --page details |
    Set-Content -Encoding utf8 (Join-Path $AnalysisDir 'details.txt')
if ($LASTEXITCODE -ne 0) { throw 'NCU details export failed' }
ncu --import "$Prefix.ncu-rep" --page raw --csv |
    Set-Content -Encoding utf8 (Join-Path $AnalysisDir 'raw.csv')
if ($LASTEXITCODE -ne 0) { throw 'NCU CSV export failed' }
```

按实际 launch/NVTX 范围添加过滤或 skip，确认报告中名称和输入匹配。
需要更多计数器时，用已枚举的 section 或 `--set full` 另存一份；需要源码归因时，
确认 `SourceCounters` 等 section 存在再采集。`PmSampling`、warp-state sampling 的
可用性依 GPU、NCU 和采集模式而定；不强制所有设备都采两套 full+source。

来自 [Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) 的
replay、cache control、clock control 和 sampling 条件都应随采集记录。短 kernel 的
采样不足应标明，不把缺失采样解释成没有 stall。NCU 的估计加速不是测得的收益。

## 三架构的诊断侧重点

| 目标 | 先看什么 | 不能据此直接下的结论 |
|---|---|---|
| SM89 | grid/tail、register/spill、shared bank conflict、DRAM/L2、MMA 与 reduction 代价 | Tensor Core 占用低就一定需要更大 tile |
| SM90 | TMA producer 与 WGMMA consumer、barrier 等待、stage、epilogue、cluster occupancy | async 指令存在就已实现有效重叠 |
| SM120 | 实际 `mma.sync` 路径、TMA 单 CTA、scale/pack 访存、register、低精度附加成本 | B200 的 TMEM/`tcgen05` 指标可衡量此实现 |

诊断顺序：先确认 launch 和资源，再结合吞吐、访存、依赖 stall 和时间分布解释。
long scoreboard 不能单独证明 DRAM 带宽饱和；低 occupancy 也不单独证明 occupancy 是瓶颈。
用至少能互相印证的测量提出假设，再通过下一候选验证。

## 分析与报告

可复用仓库 [run_ncu.py](../../../../scripts/profile/run_ncu.py)、
[analyze_report.py](../../../../scripts/profile/analyze_report.py)、
[ncu_utils.py](../../../../scripts/profile/ncu_utils.py)。先读当前参数和 metric 处理，
不能假定既有 SM89 工具已完成 SM90/SM120 适配。

需要 `ncu_report` 时用本机 Nsight Compute 的 `extras/python`，枚举 report 的全部
range/action 和实际 `metric_names()`。缺失 metric 记为 unavailable；保留单位，不将缺失
替换成 0。CLI CSV 和 Python API 可按任务选用，不将某个模块的缺失误判为无报告可读。

报告写清 parity 日志、GPU/版本、输入与实际 kernel、普通 benchmark、计数器值及单位、
推断、验证推断的下一步，以及 `.ncu-rep` / CSV 的路径。有计数器权限错误或版本不支持
时记录原始错误，交付已有有效证据并标注缺项，不凭空生成 NCU 结论。
