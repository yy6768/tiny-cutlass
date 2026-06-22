#!/usr/bin/env python3
"""ASCII 绘制 PM sampling 时间序列，揭示 tail effect 和 pipeline bubble。

PM sampling metrics（`pmsampling:` 前缀）的 per-instance 值构成时序数据，
绘制后可看到：flat-high → clean-drop（理想）、gradual tail、sawtooth 等模式。

产出物：<run-dir>/analysis/pm_timeline_plots.txt

用法：
    python scripts/profile/plot_timeline.py --run-dir profile/myrun \\
        --report profile/myrun/reports/full_baseline.ncu-rep --tag baseline
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ncu_utils import load_action, per_instance_values  # noqa: E402


DEFAULT_METRICS = [
    "pmsampling:smsp__warps_issue_stalled_long_scoreboard.avg",
    "pmsampling:smsp__warps_issue_stalled_short_scoreboard.avg",
    "pmsampling:smsp__warps_issue_stalled_wait.avg",
    "pmsampling:smsp__warps_issue_stalled_dispatch_stall.avg",
    "pmsampling:smsp__warps_issue_stalled_math_pipe_throttle.avg",
    "pmsampling:smsp__warps_issue_stalled_mio_throttle.avg",
    "pmsampling:sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "pmsampling:sm__warps_active.avg.pct_of_peak_sustained_active",
    "pmsampling:dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "pmsampling:l1tex__throughput.avg.pct_of_peak_sustained_active",
]


def ascii_plot(vals: list, label: str, max_rows: int = 20, max_cols: int = 80) -> list[str]:
    """将时序数据渲染为 ASCII 图。"""
    if not vals:
        return [f"{label}: no data"]

    vals = [v if v is not None else 0.0 for v in vals]

    # 裁剪首尾零值
    lead = 0
    for v in vals:
        if v > 0:
            break
        lead += 1
    trail = 0
    for v in reversed(vals):
        if v > 0:
            break
        trail += 1
    active = vals[lead:len(vals) - trail] if trail else vals[lead:]
    n = len(active)
    if n == 0:
        return [f"{label}: all zero"]

    # 分桶到 max_cols 列
    ncols = min(max_cols, n)
    bucket_size = max(1, n // ncols)
    buckets = []
    for c in range(ncols):
        s = c * bucket_size
        e = min(n, (c + 1) * bucket_size)
        chunk = active[s:e]
        buckets.append(sum(chunk) / len(chunk) if chunk else 0.0)
    mx = max(buckets) if buckets else 1.0
    if mx == 0:
        mx = 1.0

    lines = [
        f"\n{label}",
        f"  (n={n} active samples, leading_zero={lead}, trailing_zero={trail}, max={mx:.3g})",
    ]
    for r in range(max_rows, 0, -1):
        threshold = mx * r / max_rows
        row = "".join("#" if b >= threshold else " " for b in buckets)
        lines.append(f"  {threshold:8.2g} | {row}")
    lines.append("  " + " " * 10 + "-" * len(buckets))
    lines.append("  " + " " * 10 + " (time →)")
    return lines


def main():
    ap = argparse.ArgumentParser(description="ASCII PM-sampling timeline plots。")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, action="append", required=True)
    ap.add_argument("--tag", type=str, action="append", required=True)
    ap.add_argument("--metric", type=str, action="append", default=None,
                    help="覆盖默认 metric 列表")
    ap.add_argument("--rows", type=int, default=20, help="ASCII 图行数")
    ap.add_argument("--cols", type=int, default=80, help="ASCII 图列数")
    args = ap.parse_args()

    if len(args.report) != len(args.tag):
        ap.error("--report 和 --tag 数量必须相同")

    metrics = args.metric or DEFAULT_METRICS
    analysis_dir = args.run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    out_lines = []
    for rep, tag in zip(args.report, args.tag):
        if not rep.exists():
            print(f"[skip] {rep} 不存在", file=sys.stderr)
            continue
        action = load_action(rep)
        out_lines.append(f"\n{'=' * 60}\n{tag}: {rep.name}\n{'=' * 60}")
        for m in metrics:
            vals = per_instance_values(action, m)
            if vals is None:
                out_lines.append(f"\n{m}: no instances")
                continue
            out_lines.extend(ascii_plot(vals, m, args.rows, args.cols))

    out_path = analysis_dir / "pm_timeline_plots.txt"
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
