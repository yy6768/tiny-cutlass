from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


DEFAULT_TENSORRT_ROOT = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fp16", "fp8"], default="fp16")
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--build-dir", type=Path, default=Path("build/conv-fused"))
    parser.add_argument("--tensorrt-root", type=Path, default=DEFAULT_TENSORRT_ROOT)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--output-channels", type=int, default=16)
    parser.add_argument("--duration", type=int, default=3)
    parser.add_argument("--warm-up-ms", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--avg-runs", type=int, default=10)
    parser.add_argument("--ours-warmup", type=int, default=20)
    parser.add_argument("--ours-iterations", type=int, default=100)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--dump-output", action="store_true")
    parser.add_argument("--skip-output-export", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--skip-ours", action="store_true")
    args = parser.parse_args()
    if args.artifact_dir is None:
        args.artifact_dir = Path(
            "build/conv-fused/fp8-reference"
            if args.mode == "fp8"
            else "build/conv-fused/reference"
        )
    return args


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir.resolve()
    build_dir = args.build_dir.resolve()
    trtexec = args.tensorrt_root / "bin" / "trtexec.exe"
    tensorrt_lib_dir = args.tensorrt_root / "lib"
    if args.mode == "fp8":
        ours_exe = build_dir / "tests" / "conv-fused" / "conv1x1_relu_conv1x1_relu_trt.exe"
        onnx_name = "conv1x1_relu_conv1x1_relu_reference.onnx"
    else:
        ours_exe = build_dir / "tests" / "conv-fused" / "conv_relu_pool_trt.exe"
        onnx_name = "conv_relu_pool_reference.onnx"

    onnx_path = artifact_dir / onnx_name
    input_path = artifact_dir / "input_nhwc.bin"

    required_paths = [tensorrt_lib_dir, onnx_path, input_path]
    if not args.skip_reference:
        required_paths.append(trtexec)
    if not args.skip_ours:
        required_paths.append(ours_exe)

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    env = os.environ.copy()
    env["PATH"] = f"{tensorrt_lib_dir}{os.pathsep}{env.get('PATH', '')}"

    if not args.skip_reference:
        command = [
            str(trtexec),
            f"--onnx={onnx_name}",
            "--loadInputs=input_nhwc:input_nhwc.bin",
            f"--duration={args.duration}",
            f"--warmUp={args.warm_up_ms}",
            f"--iterations={args.iterations}",
            f"--avgRuns={args.avg_runs}",
            "--exportTimes=trt_reference_times.json",
            "--exportProfile=trt_reference_profile.json",
            "--dumpProfile",
        ]

        if args.mode == "fp8":
            command.append("--fp8")
            command.append("--fp16")
        elif not args.fp32:
            command.append("--fp16")
        if args.skip_inference:
            command.append("--skipInference")
        if not args.skip_output_export:
            command.append("--exportOutput=trt_reference_output.json")
            if args.dump_output:
                command.append("--dumpOutput")

        print(" ".join(command))
        subprocess.run(command, cwd=artifact_dir, env=env, check=True)

    if not args.skip_ours:
        ours_command = [
            str(ours_exe),
            "--artifact-dir",
            str(artifact_dir),
            "--warmup",
            str(args.ours_warmup),
            "--iterations",
            str(args.ours_iterations),
        ]
        if args.mode == "fp8":
            ours_command.extend(
                [
                    "--batch",
                    str(args.batch),
                    "--height",
                    str(args.height),
                    "--width",
                    str(args.width),
                    "--channels",
                    str(args.channels),
                    "--hidden-channels",
                    str(args.hidden_channels),
                    "--output-channels",
                    str(args.output_channels),
                ]
            )
        print(" ".join(ours_command))
        subprocess.run(ours_command, cwd=build_dir, env=env, check=True)


if __name__ == "__main__":
    main()
