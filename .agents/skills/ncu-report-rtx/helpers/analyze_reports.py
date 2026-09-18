#!/usr/bin/env python3
"""Inventory actions and extract raw metric records without assuming a GPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ncu_utils import (
    KEY_METRICS, add_selection_arguments, dump_all_metrics, iter_actions,
    metric_record, report_api, rule_results, select_action, validate_inputs,
)


def collect(path, action, ri, ai, tag, analysis):
    dump_all_metrics(action, analysis / f"metrics_all_{tag}.json")
    metrics = {name: metric_record(action, name) for name in KEY_METRICS}
    result = {
        "report": str(path.resolve()), "range_index": ri, "action_index": ai,
        "kernel": action.name(), "metrics": metrics,
        "rules": rule_results(action),
        "note": "Raw API values/units; verify scaling with CLI CSV before conversions.",
    }
    (analysis / f"metrics_key_{tag}.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    lines = [f"{tag}: {action.name()} (range {ri}, action {ai})", result["note"], ""]
    for name, record in metrics.items():
        lines.append(f"{name} = {record['value']} [{record['unit']}] ({record['status']})")
    (analysis / f"metrics_key_{tag}.txt").write_text("\n".join(lines), encoding="utf-8")
    available = sum(item["status"] == "available" for item in metrics.values())
    print(f"{tag}: {available}/{len(metrics)} candidates available; range {ri}, action {ai}")
    return result


def compare(collected, analysis):
    if len(collected) < 2:
        return
    lines = [
        "Raw metrics only; validate workload/device/capture equivalence separately.",
        "No automatic speedup or unit conversion. Cross-device peak denominators differ.",
        "Metric\t" + "\t".join(collected),
    ]
    for name in KEY_METRICS:
        records = [entry["metrics"][name] for entry in collected.values()]
        units = {r["unit"] for r in records if r["status"] == "available"}
        values = [
            f"{r['value']} [{r['unit']}]" if r["status"] == "available"
            else f"N/A ({r['status']})" for r in records
        ]
        if len(units) > 1:
            values = ["INCOMPATIBLE UNITS: " + value for value in values]
        lines.append(name + "\t" + "\t".join(values))
    path = analysis / f"compare_{'_vs_'.join(collected)}.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Comparison: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--tag", action="append")
    parser.add_argument("--run-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list-actions", action="store_true")
    mode.add_argument("--all-actions", action="store_true")
    add_selection_arguments(parser)
    args = parser.parse_args()
    if (args.all_actions or args.list_actions) and (
        args.range_index is not None or args.action_index is not None
    ):
        parser.error("Inventory/all-actions cannot be combined with action selectors")
    if args.list_actions:
        for path in args.report:
            if not path.is_file():
                parser.error(f"Report does not exist: {path}")
            report = report_api().load_report(str(path))
            inventory = [
                {"range_index": ri, "action_index": ai, "kernel": action.name(),
                 "metric_count": len(action.metric_names())}
                for ri, ai, action in iter_actions(report)
            ]
            print(json.dumps({"report": str(path), "actions": inventory}, indent=2))
        return
    if not args.run_dir or not args.tag:
        parser.error("--run-dir and one --tag per report are required for extraction")
    validate_inputs(parser, args)
    analysis = args.run_dir / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    collected = {}
    for path, tag in zip(args.report, args.tag):
        report = report_api().load_report(str(path))
        actions = list(iter_actions(report)) if args.all_actions else [
            select_action(report, args.range_index, args.action_index)
        ]
        if not actions:
            raise ValueError(f"No actions in {path}")
        for ri, ai, action in actions:
            label = f"{tag}_r{ri}_a{ai}" if args.all_actions else tag
            if label in collected:
                raise ValueError(f"Duplicate output label: {label}")
            collected[label] = collect(path, action, ri, ai, label, analysis)
    # --all-actions inventories kernels; it does not imply they are comparable.
    if not args.all_actions:
        compare(collected, analysis)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(str(exc))
