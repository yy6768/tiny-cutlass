from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=Path("build/conv-fused/reference"))
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--candidate", type=Path, action="append", default=None)
    parser.add_argument("--output-name", default="output_nhwc")
    parser.add_argument("--mae", type=float, default=1.0e-3)
    parser.add_argument("--max-abs", type=float, default=2.0e-2)
    return parser.parse_args()


def numeric_array(value: Any, expected_size: int) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size == expected_size:
        return array
    return None


def find_output(value: Any, output_name: str, expected_size: int) -> np.ndarray | None:
    direct = numeric_array(value, expected_size)
    if direct is not None:
        return direct

    if isinstance(value, dict):
        if output_name in value:
            found = find_output(value[output_name], output_name, expected_size)
            if found is not None:
                return found

        name = value.get("name") or value.get("binding") or value.get("tensorName")
        if name == output_name:
            for key in ["values", "data", "output", "raw"]:
                if key in value:
                    found = find_output(value[key], output_name, expected_size)
                    if found is not None:
                        return found

        for nested in value.values():
            found = find_output(nested, output_name, expected_size)
            if found is not None:
                return found

    if isinstance(value, list):
        for nested in value:
            found = find_output(nested, output_name, expected_size)
            if found is not None:
                return found

    return None


def load_candidate(path: Path, reference: np.ndarray, output_name: str) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        candidate = np.load(path)
    elif suffix == ".bin":
        candidate = np.fromfile(path, dtype=reference.dtype)
    else:
        with path.open("r", encoding="utf-8") as handle:
            candidate_json = json.load(handle)
        candidate = find_output(candidate_json, output_name, reference.size)
        if candidate is None:
            raise RuntimeError(
                f"could not find {output_name} with {reference.size} values in {path}"
            )

    if candidate.size != reference.size:
        raise RuntimeError(
            f"candidate has {candidate.size} values, expected {reference.size}: {path}"
        )
    return candidate.reshape(reference.shape).astype(np.float32)


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir
    reference_path = args.reference or artifact_dir / "output_nhwc.npy"
    candidate_paths = args.candidate or [
        artifact_dir / "trt_reference_output.json",
        artifact_dir / "ours_output.bin",
    ]

    reference_raw = np.load(reference_path)
    reference = reference_raw.astype(np.float32)
    for candidate_path in candidate_paths:
        candidate = load_candidate(candidate_path, reference_raw, args.output_name)
        diff = np.abs(reference - candidate)
        mae = float(diff.mean())
        max_abs = float(diff.max())

        print(f"shape={reference.shape}")
        print(f"candidate={candidate_path}")
        print(f"mae={mae}")
        print(f"max_abs={max_abs}")

        if mae > args.mae or max_abs > args.max_abs:
            raise SystemExit(
                f"verification failed: mae {mae} > {args.mae} or max_abs {max_abs} > {args.max_abs}"
            )


if __name__ == "__main__":
    main()
