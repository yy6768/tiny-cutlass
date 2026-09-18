#!/usr/bin/env python3
"""Plot collected PM series without zero trimming or dropping the tail."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ncu_utils import (
    add_selection_arguments, correlated_values, load_report, validate_inputs,
)


def numeric(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def bucket_series(values, columns, timestamps=None):
    """All samples are retained. Empty/missing buckets remain unknown."""
    if columns < 1:
        raise ValueError("columns must be positive")
    if not values:
        return []
    count = min(columns, len(values))
    bins = [[] for _ in range(count)]
    timed = (timestamps is not None and len(timestamps) == len(values)
             and all(numeric(t) for t in timestamps)
             and all(a <= b for a, b in zip(timestamps, timestamps[1:]))
             and timestamps[-1] > timestamps[0])
    for index, value in enumerate(values):
        if timed:
            col = min(count - 1, int((timestamps[index] - timestamps[0]) /
                                    (timestamps[-1] - timestamps[0]) * count))
        else:
            col = min(count - 1, index * count // len(values))
        bins[col].append(value)
    # Missing samples make their bucket unknown, rather than silently biasing a mean.
    return [sum(bucket) / len(bucket) if bucket and all(numeric(v) for v in bucket)
            else None for bucket in bins]


def ascii_plot(values, label, max_rows=12, max_cols=80, timestamps=None):
    if max_rows < 1 or max_cols < 1:
        raise ValueError("rows and columns must be positive")
    if not values:
        return [f"{label}: N/A (no samples)"]
    timed = (timestamps is not None and len(timestamps) == len(values)
             and all(numeric(t) for t in timestamps)
             and all(a <= b for a, b in zip(timestamps, timestamps[1:]))
             and timestamps[-1] > timestamps[0])
    buckets = bucket_series(values, max_cols, timestamps if timed else None)
    peak = max([v for v in buckets if v is not None] + [0.0])
    scale = peak if peak > 0 else 1.0
    missing = sum(not numeric(v) for v in values)
    lines = [f"\n{label}: n={len(values)}, missing={missing}, peak_bucket={peak:g}"]
    if timed:
        lines.append(f"timestamp ns: {timestamps[0]} .. {timestamps[-1]} (equal-time bins)")
    else:
        lines.append("sample index axis; no usable timestamp span, not an elapsed-time plot")
    for row in range(max_rows, 0, -1):
        threshold = scale * row / max_rows
        chars = "".join("?" if value is None else "#" if value >= threshold else " "
                        for value in buckets)
        lines.append(f"{threshold:10.3g} |{chars}|")
    lines.append("           +" + "-" * len(buckets) + "+")
    lines.append("Zeros retained; ? = empty/missing bucket; no cross-pass alignment.")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--tag", action="append", required=True)
    parser.add_argument("--metric", action="append", help="Exact collected pmsampling: name.")
    parser.add_argument("--rows", type=int, default=12)
    parser.add_argument("--cols", type=int, default=80)
    add_selection_arguments(parser)
    args = parser.parse_args()
    validate_inputs(parser, args)
    if args.rows < 1 or args.cols < 1:
        parser.error("--rows and --cols must be positive")
    if args.metric and any(not name.startswith("pmsampling:") for name in args.metric):
        parser.error("Only pmsampling: metrics are timelines")
    analysis = args.run_dir / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    for path, tag in zip(args.report, args.tag):
        report, action = load_report(path, args.range_index, args.action_index)
        names = args.metric or sorted(n for n in action.metric_names()
                                     if n.startswith("pmsampling:"))
        lines = [f"{tag}: {path} / {action.name()}",
                 "Raw PM values; other GPU activity and replay alignment may affect samples."]
        archive = {}
        if not names:
            lines.append("N/A: no PM metrics collected.")
        for name in names:
            samples = correlated_values(action, name)
            timestamps = [time for time, _ in samples]
            values = [value for _, value in samples]
            archive[name] = {"samples": samples, "correlation_unit": "ns"}
            lines.extend(ascii_plot(values, name, args.rows, args.cols, timestamps))
        (analysis / f"pm_timeline_{tag}.txt").write_text("\n".join(lines), encoding="utf-8")
        (analysis / f"pm_samples_{tag}.json").write_text(
            json.dumps(archive, indent=2, default=str), encoding="utf-8"
        )
        print(f"{tag}: {len(names)} PM metric series")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(str(exc))
