#!/usr/bin/env python3
"""从 .ncu-rep 文件提取关键 metrics 并可选做 A/B 对比。

产出物放在 <run-dir>/analysis/ 下：
    metrics_all_<tag>.json    — 全量 metric 归档
    metrics_key_<tag>.txt/json — 精选关键 metrics
    compare_<tag1>_vs_<tag2>.txt — 多报告对比（≥2 个 report 时生成）

用法：
    python scripts/profile/analyze_report.py --run-dir profile/myrun \\
        --report profile/myrun/reports/full_baseline.ncu-rep --tag baseline

    # 多报告对比
    python scripts/profile/analyze_report.py --run-dir profile/myrun \\
        --report profile/myrun/reports/full_v1.ncu-rep --tag v1 \\
        --report profile/myrun/reports/full_v2.ncu-rep --tag v2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ncu_utils import SM89_KEY_METRICS, dump_all_metrics, load_action, safe  # noqa: E402


def collect(report_path: Path, tag: str, analysis_dir: Path) -> dict:
    action = load_action(report_path)
    print(f"[{tag}] {report_path.name}: {len(action.metric_names())} metrics, kernel {action.name()}")

    # 全量归档
    n = dump_all_metrics(action, analysis_dir / f"metrics_all_{tag}.json")
    print(f"  -> metrics_all_{tag}.json ({n} metrics)")

    # 精选 key metrics
    key = {m: safe(action, m) for m in SM89_KEY_METRICS}
    key["__kernel_name__"] = action.name()

    (analysis_dir / f"metrics_key_{tag}.json").write_text(
        json.dumps(key, indent=2, default=str), encoding="utf-8"
    )
    with open(analysis_dir / f"metrics_key_{tag}.txt", "w", encoding="utf-8") as f:
        f.write(f"===== {tag} =====\nKernel: {action.name()}\n\n")
        for m, v in key.items():
            if m.startswith("__"):
                continue
            f.write(f"{m:95s} = {v}\n")
    print(f"  -> metrics_key_{tag}.{{json,txt}}")
    return key


def compare(collected: dict, analysis_dir: Path):
    tags = list(collected.keys())
    if len(tags) < 2:
        return
    out_path = analysis_dir / f"compare_{'_vs_'.join(tags)}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        col_w = max(20, max(len(t) for t in tags) + 2)
        f.write(f"{'Metric':<95}")
        for t in tags:
            f.write(f"{t:>{col_w}}")
        f.write("\n" + "-" * (95 + col_w * len(tags)) + "\n")
        for m in SM89_KEY_METRICS:
            f.write(f"{m:<95}")
            for t in tags:
                v = collected[t].get(m, "N/A")
                if isinstance(v, (int, float)):
                    v = f"{v:.4g}"
                f.write(f"{str(v):>{col_w}}")
            f.write("\n")
    print(f"compare -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="提取 NCU 关键 metrics 并对比。")
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="profile run 目录，输出写入 <run-dir>/analysis/")
    ap.add_argument("--report", type=Path, action="append", required=True,
                    help=".ncu-rep 文件路径，可多次传入")
    ap.add_argument("--tag", type=str, action="append", required=True,
                    help="每个 report 的短标签，数量必须与 --report 匹配")
    args = ap.parse_args()

    if len(args.report) != len(args.tag):
        ap.error("--report 和 --tag 数量必须相同")

    analysis_dir = args.run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    collected = {}
    for rep, tag in zip(args.report, args.tag):
        if not rep.exists():
            print(f"[skip] {rep} 不存在", file=sys.stderr)
            continue
        collected[tag] = collect(rep, tag, analysis_dir)

    compare(collected, analysis_dir)


if __name__ == "__main__":
    main()
