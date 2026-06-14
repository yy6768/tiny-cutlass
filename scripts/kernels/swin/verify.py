from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--config", default="Release")
    return parser.parse_args()


def executable(build_dir: Path, config: str, name: str) -> Path:
    suffix = ".exe" if sys.platform.startswith("win") else ""
    return build_dir / "csrc" / "swin" / config / f"{name}{suffix}"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


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


if __name__ == "__main__":
    main()
