#!/usr/bin/env python3
"""Aggregate collected PC samples, separating stalls from scheduler selection."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from ncu_utils import (
    add_selection_arguments, load_report, per_pc_values, pc_to_source_line,
    validate_inputs,
)

PREFIX = "smsp__pcsamp_warps_issue_stalled_"
SCHEDULER_STATES = {"selected", "not_selected"}


def collect_per_pc(action, sample_scope="all"):
    per_pc = defaultdict(lambda: defaultdict(int))
    available = sorted(name for name in action.metric_names()
                       if name.startswith(PREFIX)
                       and name.endswith("_not_issued") == (sample_scope == "not-issued"))
    for name in available:
        for pc, value in per_pc_values(action, name):
            if isinstance(value, (int, float)) and value > 0:
                state = name[len(PREFIX):]
                if sample_scope == "not-issued":
                    state = state.removesuffix("_not_issued")
                per_pc[pc][state] += value
    return per_pc, available


def aggregate_by_source_line(action, per_pc):
    result = defaultdict(lambda: defaultdict(int))
    for pc, counts in per_pc.items():
        location = pc_to_source_line(action, pc)
        for name, count in counts.items():
            result[location][name] += count
    return result


def write_report(action, per_line, available, path, tag, top, sample_scope="all"):
    lines = [
        f"{tag}: {action.name()}",
        "Statistical PC sample counts, not elapsed cycles or causal load attribution.",
        "selected and not_selected are excluded from stall totals.",
        f"Sample scope: {sample_scope}; all-state and not-issued subset counts are never added.",
        f"Collected state metrics: {len(available)}",
    ]
    if not available:
        lines.append("N/A: no PC state metrics collected; request supported SourceCounters.")
    elif not per_line:
        lines.append("No positive correlated PC samples; not evidence of a stall-free kernel.")
    ranked = sorted(
        per_line.items(),
        key=lambda item: -sum(v for n, v in item[1].items() if n not in SCHEDULER_STATES),
    )
    lines.append("\nTop source/PC stall samples:")
    for (filename, line), counts in ranked[:top]:
        stalled = sum(v for n, v in counts.items() if n not in SCHEDULER_STATES)
        breakdown = ", ".join(f"{n}={v:g}" for n, v in sorted(counts.items())
                              if n not in SCHEDULER_STATES)
        lines.append(f"{stalled:g}\t{filename}:{line}\t{breakdown}")
    lines.append("\nScheduler selection samples (separate):")
    for state in sorted(SCHEDULER_STATES):
        selected = sorted(per_line.items(), key=lambda item: -item[1].get(state, 0))
        for (filename, line), counts in selected[:top]:
            if counts.get(state, 0):
                lines.append(f"{state}={counts[state]:g}\t{filename}:{line}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--tag", action="append", required=True)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--sample-scope", choices=("all", "not-issued"), default="all",
                        help="Use all warp-state samples OR the not-issued subset, never both.")
    add_selection_arguments(parser)
    args = parser.parse_args()
    validate_inputs(parser, args)
    if args.top < 1:
        parser.error("--top must be positive")
    analysis = args.run_dir / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    for path, tag in zip(args.report, args.tag):
        report, action = load_report(path, args.range_index, args.action_index)
        per_pc, available = collect_per_pc(action, args.sample_scope)
        per_line = aggregate_by_source_line(action, per_pc)
        output = analysis / f"stall_hotspots_{tag}.txt"
        write_report(action, per_line, available, output, tag, args.top, args.sample_scope)
        print(f"{tag}: {len(per_line)} source/PC entries -> {output}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(str(exc))
