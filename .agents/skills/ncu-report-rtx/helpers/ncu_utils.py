"""Typed, action-aware Nsight Compute helpers for SM80/SM86/SM89/SM120.

Candidate metrics are not guaranteed on any device. NVIDIA's report API is
loaded only when a report is opened; --help and plotting tests work without it.
"""
from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import sys
from pathlib import Path


def _module_paths():
    explicit = os.environ.get("NCU_PYTHON_PATH")
    if explicit:
        return [Path(explicit)]
    paths = []
    exe = shutil.which("ncu") or shutil.which("ncu.exe")
    if exe:
        base = Path(exe).resolve().parent
        paths.extend([base / "extras/python", base.parent / "extras/python"])
    roots = [
        Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "NVIDIA Corporation",
        Path(os.environ.get("ProgramW6432", "C:/Program Files")) / "NVIDIA Corporation",
        Path("/usr/local"), Path("/opt/nvidia"), Path("/opt"),
    ]
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        roots.append(Path(cuda_path))
    for root in roots:
        if root.is_dir():
            for pattern in ("Nsight Compute */extras/python", "nsight-compute*/extras/python",
                            "cuda*/nsight-compute*/extras/python"):
                paths.extend(sorted(root.glob(pattern), reverse=True))
    return list(dict.fromkeys(paths))


def report_api():
    explicit = os.environ.get("NCU_PYTHON_PATH")
    if not explicit:
        try:
            return importlib.import_module("ncu_report")
        except ImportError:
            pass
    errors = []
    for candidate in _module_paths():
        if not (candidate / "ncu_report.py").is_file():
            continue
        sys.path.insert(0, str(candidate))
        try:
            return importlib.import_module("ncu_report")
        except (ImportError, OSError) as exc:
            errors.append(f"{candidate}: {exc}")
            sys.modules.pop("ncu_report", None)
        finally:
            sys.path.remove(str(candidate))
    detail = "; ".join(errors) or "No matching extras/python module found."
    raise RuntimeError(
        "Cannot import NVIDIA ncu_report. Set NCU_PYTHON_PATH to a compatible "
        "Nsight Compute extras/python directory or export CSV with ncu. " + detail
    )


def iter_actions(report):
    for ri in range(report.num_ranges()):
        rng = report.range_by_idx(ri)
        for ai in range(rng.num_actions()):
            yield ri, ai, rng.action_by_idx(ai)


def select_action(report, range_index=None, action_index=None):
    if (range_index is not None and range_index < 0) or (
        action_index is not None and action_index < 0
    ):
        raise ValueError("Range and action indices must be nonnegative.")
    matches = [
        (ri, ai, action) for ri, ai, action in iter_actions(report)
        if (range_index is None or ri == range_index)
        and (action_index is None or ai == action_index)
    ]
    if len(matches) != 1:
        inventory = ", ".join(f"{ri}:{ai} {a.name()}" for ri, ai, a in iter_actions(report))
        raise ValueError(
            f"Selection matched {len(matches)} actions; choose --range-index and "
            f"--action-index (or list/all actions in analyzer). Available: {inventory or 'none'}"
        )
    return matches[0]


def load_report(path, range_index=None, action_index=None):
    """Return (report, selected action); caller must keep report alive."""
    report = report_api().load_report(str(path))
    _, _, action = select_action(report, range_index, action_index)
    return report, action


def safe(action, name, default=None):
    try:
        value = action[name].value()
        return default if value is None else value
    except Exception:
        return default


def safe_many(action, names, default=None):
    return {name: safe(action, name, default) for name in names}


def metric_record(action, name):
    record = {"name": name, "value": None, "unit": None, "status": "not_collected"}
    if name not in action.metric_names():
        return record
    try:
        metric = action[name]
        record["unit"] = metric.unit()
        record["value"] = metric.value()
        record["status"] = "available" if record["value"] is not None else "unavailable"
    except Exception as exc:
        record.update(status="error", error=str(exc))
    return record


def metric_value_at(metric, index):
    """value(index) dispatches using the instance kind and preserves missing data."""
    try:
        return metric.value(index)
    except Exception:
        return None


def per_instance_values(action, name):
    try:
        metric = action[name]
        count = metric.num_instances()
        return [metric_value_at(metric, i) for i in range(count)] if count else None
    except Exception:
        return None


def correlated_values(action, name):
    """Return (correlation, value) instances; missing correlation stays None."""
    try:
        metric = action[name]
        ids = metric.correlation_ids() if metric.has_correlation_ids() else None
        return [(metric_value_at(ids, i) if ids is not None else None,
                 metric_value_at(metric, i)) for i in range(metric.num_instances())]
    except Exception:
        return []


def per_pc_values(action, name):
    return [(pc, value) for pc, value in correlated_values(action, name)
            if pc is not None]


def pc_to_source_line(action, pc):
    try:
        source = action.source_info(pc)
        if source is not None:
            return source.file_name(), source.line()
    except Exception:
        pass
    return f"<unmapped PC 0x{int(pc):x}>", 0


def dump_all_metrics(action, outfile):
    records = [metric_record(action, name) for name in sorted(action.metric_names())]
    Path(outfile).write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    return len(records)


def rule_results(action):
    try:
        return list(action.rule_results_as_dicts())
    except Exception:
        return []


def rule_speedups(action):
    result = []
    for rule in rule_results(action):
        try:
            estimate = float(rule.get("estimated_speedup_pct") or 0)
        except (TypeError, ValueError):
            estimate = 0.0
        result.append((estimate, rule.get("rule_name", "?"),
                       rule.get("message_for_display", "?")))
    return sorted(result, key=lambda item: -item[0])


def add_selection_arguments(parser):
    parser.add_argument("--range-index", type=int, help="Explicit report range index.")
    parser.add_argument("--action-index", type=int, help="Explicit action index in range.")


def validate_inputs(parser, args):
    if len(args.report) != len(args.tag):
        parser.error("--report and --tag counts must match")
    if len(set(args.tag)) != len(args.tag):
        parser.error("Tags must be unique")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tag) for tag in args.tag):
        parser.error("Tags must be simple filename components (letters/digits/_.-)")
    for report in args.report:
        if not report.is_file():
            parser.error(f"Report does not exist: {report}")


# Extraction candidates: intersect with actual collected names, retain missing
# entries as not_collected, and verify support through the saved device query.
# No active/elapsed, ratio/percent, or dtype-specific substitutions are made.
KEY_METRICS = [
    "device__attribute_display_name",
    "device__attribute_compute_capability_major",
    "device__attribute_compute_capability_minor",
    "device__attribute_max_blocks_per_multiprocessor",
    "device__attribute_max_shared_memory_per_multiprocessor",
    "device__attribute_max_shared_memory_per_block_optin",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__cycles_active.min",
    "sm__cycles_active.avg",
    "sm__cycles_active.max",
    "launch__shared_mem_per_block_static",
    "launch__shared_mem_per_block_dynamic",

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
    # SOL
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    # Occupancy
    "sm__maximum_warps_per_active_cycle_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.per_cycle_active",
    "sm__warps_active.max.per_cycle_active",
    "sm__warps_active.min.per_cycle_active",
    "smsp__warps_active.avg.per_cycle_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__warps_eligible.max.per_cycle_active",
    # IPC
    "sm__inst_executed.avg.per_cycle_active",
    "smsp__issue_active.avg.per_cycle_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed.avg",
    # Compute pipes
    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_adu.avg.pct_of_peak_sustained_active",
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
    "l1tex__t_sector_pipe_lsu_mem_global_op_st_hit_rate.pct",
    # Memory instruction counts
    "smsp__sass_inst_executed_op_global_ld.sum",
    "smsp__sass_inst_executed_op_global_st.sum",
    "smsp__sass_inst_executed_op_local_ld.sum",
    "smsp__sass_inst_executed_op_local_st.sum",
    "smsp__sass_inst_executed_op_shared.sum",
    "smsp__sass_inst_executed_op_shared_ld.sum",
    "smsp__sass_inst_executed_op_shared_st.sum",
    # Sectors / requests (for coalescing analysis)
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio",
    # Stall reasons — aggregate ratios
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_membar_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_drain_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_misc_per_issue_active.ratio",
    # Per-PC sampling candidates (request supported SourceCounters)
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
