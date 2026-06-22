r"""共享工具函数：加载 Nsight Compute 报告、安全读取 metric。

Usage:
    from ncu_utils import load_report, safe, dump_all_metrics

调用者需要将 ncu_report 放到 PYTHONPATH 上，例如：
    set PYTHONPATH=%PYTHONPATH%;C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.2.0\extras\python

如果未设置，本模块会尝试常见路径自动定位。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


# --- 定位 ncu_report 模块 ---------------------------------------------------

def _locate_ncu_report() -> str | None:
    """尝试常见安装路径定位 ncu_report Python 模块。"""
    candidates: list[str] = []

    # Windows 常见位置
    if sys.platform.startswith("win"):
        nv_base = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        nv_corp = nv_base / "NVIDIA Corporation"
        if nv_corp.is_dir():
            for sub in nv_corp.glob("Nsight Compute */extras/python"):
                candidates.append(str(sub))
        # CUDA Toolkit 自带的 nsight-compute
        cuda_base = Path(os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"))
        for sub in cuda_base.parent.glob("*/nsight-compute-*/extras/python"):
            candidates.append(str(sub))
    else:
        # Linux 常见位置
        for root in ["/usr/local", "/opt/nvidia", "/opt/cuda"]:
            p = Path(root)
            if not p.is_dir():
                continue
            for sub in p.glob("cuda-*/nsight-compute-*/extras/python"):
                candidates.append(str(sub))
            for sub in p.glob("nsight-compute-*/extras/python"):
                candidates.append(str(sub))

    for c in candidates:
        if Path(c).is_dir() and (Path(c) / "ncu_report.py").exists():
            return c
    return None


try:
    import ncu_report  # noqa: F401
except ImportError:
    found = _locate_ncu_report()
    if found:
        sys.path.insert(0, found)
        import ncu_report  # noqa: F401
    else:
        raise ImportError(
            "无法导入 ncu_report。请设置 PYTHONPATH 指向 Nsight Compute 的 extras/python 目录，"
            r"例如: set PYTHONPATH=%PYTHONPATH%;C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.2.0\extras\python"
        )

import ncu_report  # noqa: E402


# --- 加载报告 ----------------------------------------------------------------

def load_report(path: str | Path):
    """加载 .ncu-rep 文件，返回 (report, action) 元组。

    action 是第一个 kernel launch 的数据对象。
    """
    r = ncu_report.load_report(str(path))
    rng = r.range_by_idx(0)
    action = rng.action_by_idx(0)
    return r, action


def load_action(path: str | Path):
    """快捷方式：只需要 action 时使用。"""
    _, action = load_report(path)
    return action


# --- 安全读取 metric ---------------------------------------------------------

def safe(action, name: str, default=None):
    """读取 metric 值，缺失或出错时返回 default。"""
    try:
        return action[name].value()
    except Exception:
        return default


def safe_many(action, names: list[str], default=None) -> dict:
    """批量读取多个 metric，返回 {name: value} 字典。"""
    return {n: safe(action, n, default) for n in names}


def metric_or_none(action, *candidates):
    """依次尝试多个 metric 名，返回第一个有效值。
    用于处理不同 GPU 代际的 metric 名差异。"""
    for n in candidates:
        v = safe(action, n, None)
        if v is not None:
            return v
    return None


# --- Per-instance 值访问 -----------------------------------------------------

def metric_value_at(m, i):
    """读取 metric 第 i 个 instance 的值（兼容不同 ValueKind）。"""
    k = m.kind()
    if k == m.ValueKind_UINT64:
        return m.as_uint64(i)
    if k in (m.ValueKind_DOUBLE, m.ValueKind_FLOAT):
        return m.as_double(i)
    if k == m.ValueKind_STRING:
        return m.as_string(i)
    try:
        return m.as_uint64(i)
    except Exception:
        try:
            return m.as_double(i)
        except Exception:
            return None


def per_instance_values(action, metric_name: str) -> list | None:
    """返回 metric 的所有 per-instance 值列表，无数据则返回 None。"""
    try:
        m = action[metric_name]
    except Exception:
        return None
    try:
        n = m.num_instances()
    except Exception:
        return None
    if n == 0:
        return None
    return [metric_value_at(m, i) for i in range(n)]


# --- 导出所有 metrics --------------------------------------------------------

def dump_all_metrics(action, outfile: str | Path) -> int:
    """将所有 metric name + value 导出为 JSON 文件，返回条目数。"""
    out = []
    for n in sorted(action.metric_names()):
        try:
            m = action[n]
            rec = {"name": n}
            try:
                rec["value"] = m.value()
            except Exception as e:
                rec["error"] = str(e)
            try:
                rec["unit"] = m.unit()
            except Exception:
                pass
            out.append(rec)
        except Exception as e:
            out.append({"name": n, "error": str(e)})
    Path(outfile).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return len(out)


# --- PC → 源码行映射 ---------------------------------------------------------

def per_pc_values(action, metric_name: str) -> list[tuple]:
    """对 source-level metric（按 PC 关联），返回 [(pc, value), ...] 列表。"""
    try:
        m = action[metric_name]
    except Exception:
        return []
    try:
        n = m.num_instances()
    except Exception:
        return []
    if n == 0 or not m.has_correlation_ids():
        return []
    cor = m.correlation_ids()
    out = []
    for i in range(n):
        try:
            pc = cor.as_uint64(i)
        except Exception:
            try:
                pc = int(cor.as_double(i))
            except Exception:
                pc = None
        try:
            v = metric_value_at(m, i)
        except Exception:
            v = 0
        out.append((pc, v))
    return out


def pc_to_source_line(action, pc) -> tuple[str, int]:
    """给定 PC 返回 (file, line)，无信息则返回 ('?', 0)。
    需要编译时使用 -lineinfo。"""
    try:
        si = action.source_info(pc)
        if si is None:
            return "?", 0
        return si.file_name(), si.line()
    except Exception:
        return "?", 0


# --- SM89 (RTX 4090 / Ada) 关键 Metrics 列表 --------------------------------
#
# 大部分 metric 名在 SM80-SM89 通用。个别 SM100 专属名称已移除。
# 使用前建议用 action.metric_names() 验证可用性。

SM89_KEY_METRICS = [
    # Launch geometry
    "launch__grid_size",
    "launch__block_size",
    "launch__grid_dim_x",
    "launch__grid_dim_y",
    "launch__grid_dim_z",
    "launch__block_dim_x",
    "launch__waves_per_multiprocessor",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
    "launch__thread_count",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "device__attribute_multiprocessor_count",
    "device__attribute_max_warps_per_multiprocessor",
    # Timing
    "gpu__time_duration.sum",
    "smsp__cycles_active.avg",
    # SOL (Speed of Light)
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    # Occupancy
    "sm__maximum_warps_per_active_cycle_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.per_cycle_active",
    "smsp__warps_active.avg.per_cycle_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    # IPC
    "sm__inst_executed.avg.per_cycle_active",
    "smsp__issue_active.avg.per_cycle_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    # Compute pipes
    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed",
    # Tensor core
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    # DRAM
    "dram__bytes_read.sum",
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum.per_second",
    "dram__bytes_write.sum",
    "dram__bytes_write.sum.pct_of_peak_sustained_elapsed",
    "dram__sectors_read.sum",
    "dram__sectors_write.sum",
    # Caches
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",
    # Sectors / requests (coalescing)
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    # Memory instruction counts
    "smsp__sass_inst_executed_op_global_ld.sum",
    "smsp__sass_inst_executed_op_global_st.sum",
    "smsp__sass_inst_executed_op_local_ld.sum",
    "smsp__sass_inst_executed_op_local_st.sum",
    "smsp__sass_inst_executed_op_shared_ld.sum",
    "smsp__sass_inst_executed_op_shared_st.sum",
    # Stall reasons — aggregate ratios
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_membar_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_drain_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio",
    # Stall reasons — per-PC (requires --set source)
    "smsp__pcsamp_sample_count",
    "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
    "smsp__pcsamp_warps_issue_stalled_short_scoreboard",
    "smsp__pcsamp_warps_issue_stalled_wait",
    "smsp__pcsamp_warps_issue_stalled_barrier",
    "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle",
    "smsp__pcsamp_warps_issue_stalled_mio_throttle",
    "smsp__pcsamp_warps_issue_stalled_lg_throttle",
    "smsp__pcsamp_warps_issue_stalled_not_selected",
    "smsp__pcsamp_warps_issue_stalled_dispatch_stall",
    "smsp__pcsamp_warps_issue_stalled_drain",
    "smsp__pcsamp_warps_issue_stalled_no_instructions",
    "smsp__pcsamp_warps_issue_stalled_selected",
    "smsp__pcsamp_warps_issue_stalled_branch_resolving",
    "smsp__pcsamp_warps_issue_stalled_membar",
]
