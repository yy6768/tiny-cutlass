from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--config", default="Release")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoint"))
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    return parser.parse_args()


def executable(build_dir: Path, config: str, name: str) -> Path:
    suffix = ".exe" if sys.platform.startswith("win") else ""
    return build_dir / "csrc" / "swin" / config / f"{name}{suffix}"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def add_file_args(base_dir: Path, files: dict[str, str | int | bool]) -> list[str]:
    args: list[str] = []
    for key, value in files.items():
        if isinstance(value, str):
            args.append(f"--{key}={base_dir / value}")
        else:
            args.append(f"--{key}={value}")
    return args


def prepare_official(checkpoint_dir: Path) -> None:
    run(
        [
            sys.executable,
            "scripts/kernels/swin/prepare_official.py",
            "--checkpoint-dir",
            str(checkpoint_dir),
        ]
    )


def run_official_cases(args: argparse.Namespace, patch_embed: Path, window: Path) -> None:
    if not args.skip_prepare:
        prepare_official(args.checkpoint_dir)

    manifest_path = args.checkpoint_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing official manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    patch = manifest["patch_embed"]
    run(
        [
            str(patch_embed),
            f"--batch_size={patch['batch_size']}",
            f"--image_size={patch['image_size']}",
            f"--in_channels={patch['in_channels']}",
            f"--input_channels_padded={patch['input_channels_padded']}",
            f"--embed_dim={patch['embed_dim']}",
            f"--patch_size={patch['patch_size']}",
            "--iterations=1",
            "--reference-check=true",
            *add_file_args(
                args.checkpoint_dir,
                {
                    "input-file": patch["input"],
                    "kernel-file": patch["kernel"],
                    "bias-file": patch["bias"],
                    "gamma-file": patch["gamma"],
                    "beta-file": patch["beta"],
                },
            ),
        ]
    )

    for stage in manifest["stages"]:
        for shift in stage["shifts"]:
            run(
                [
                    str(window),
                    f"--batch_size={manifest['batch_size']}",
                    f"--image_size={stage['image_size']}",
                    f"--window_size={stage['window_size']}",
                    f"--shift_size={shift['shift_size']}",
                    f"--head_number={stage['head_number']}",
                    f"--head_size={stage['head_size']}",
                    "--iterations=1",
                    "--reference-check=true",
                    f"--mask={str(bool(shift['use_mask'])).lower()}",
                    *add_file_args(
                        args.checkpoint_dir,
                        {
                            "input-file": stage["input"],
                            "qkv-weight-file": shift["qkv_weight"],
                            "qkv-bias-file": shift["qkv_bias"],
                            "output-weight-file": shift["output_weight"],
                            "output-bias-file": shift["output_bias"],
                            "rel-bias-file": shift["rel_bias"],
                        },
                    ),
                ]
            )


def main() -> None:
    args = parse_args()
    window = executable(args.build_dir, args.config, "swin_window")
    patch_embed = executable(args.build_dir, args.config, "swin_patch_embed")

    run(
        [
            str(patch_embed),
            "--batch_size=1",
            "--image_size=224",
            "--in_channels=3",
            "--input_channels_padded=8",
            "--embed_dim=96",
            "--patch_size=4",
            "--iterations=1",
        ]
    )

    run(
        [
            str(window),
            "--batch_size=1",
            "--image_size=14",
            "--window_size=7",
            "--shift_size=3",
            "--head_number=3",
            "--head_size=32",
            "--iterations=1",
            "--reference-check=true",
        ]
    )

    run(
        [
            str(window),
            "--batch_size=1",
            "--image_size=7",
            "--window_size=7",
            "--shift_size=0",
            "--head_number=24",
            "--head_size=32",
            "--iterations=1",
            "--reference-check=true",
            "--mask=false",
        ]
    )

    if args.official:
        run_official_cases(args, patch_embed, window)


if __name__ == "__main__":
    main()
