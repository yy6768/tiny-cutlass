from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class ConvReluPoolReference(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(
            input_channels,
            hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(
            hidden_channels,
            output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input_nhwc: torch.Tensor) -> torch.Tensor:
        x = input_nhwc.permute(0, 3, 1, 2).contiguous()
        x = self.pool0(torch.relu(self.conv0(x)))
        x = self.pool1(torch.relu(self.conv1(x)))
        return x.permute(0, 2, 3, 1).contiguous()


def save_array(output_dir: Path, name: str, array: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(array)
    npy_path = output_dir / f"{name}.npy"
    bin_path = output_dir / f"{name}.bin"
    np.save(npy_path, array)
    array.tofile(bin_path)
    return {
        "npy": npy_path.name,
        "bin": bin_path.name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("build/conv-fused/reference"))
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--output-channels", type=int, default=64)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.fp32 else torch.float16

    model = ConvReluPoolReference(
        args.channels,
        args.hidden_channels,
        args.output_channels,
    ).to(device=device, dtype=dtype)
    model.eval()

    input_nhwc = torch.randn(
        args.batch,
        args.height,
        args.width,
        args.channels,
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        output_nhwc = model(input_nhwc)

    onnx_path = args.output_dir / "conv_relu_pool_reference.onnx"
    torch.onnx.export(
        model,
        input_nhwc,
        onnx_path,
        input_names=["input_nhwc"],
        output_names=["output_nhwc"],
        opset_version=args.opset,
        do_constant_folding=True,
    )

    conv0_weight_krsc = (
        model.conv0.weight.detach().permute(0, 2, 3, 1).contiguous().cpu().numpy()
    )
    conv1_weight_krsc = (
        model.conv1.weight.detach().permute(0, 2, 3, 1).contiguous().cpu().numpy()
    )

    artifacts: dict[str, object] = {
        "model": onnx_path.name,
        "layout": {
            "input": "NHWC",
            "output": "NHWC",
            "weights": "KRSC",
        },
        "operator": "conv3x3_relu_pool_conv1x1_relu_pool",
        "seed": args.seed,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "problem": {
            "input_nhwc": [args.batch, args.height, args.width, args.channels],
            "conv0": {
                "kernel": [3, 3],
                "padding": [1, 1],
                "stride": [1, 1],
                "channels": [args.channels, args.hidden_channels],
            },
            "pool0": {"kernel": [2, 2], "stride": [2, 2]},
            "conv1": {
                "kernel": [1, 1],
                "padding": [0, 0],
                "stride": [1, 1],
                "channels": [args.hidden_channels, args.output_channels],
            },
            "pool1": {"kernel": [2, 2], "stride": [2, 2]},
            "output_nhwc": list(output_nhwc.shape),
        },
        "artifacts": {
            "input_nhwc": save_array(args.output_dir, "input_nhwc", input_nhwc.cpu().numpy()),
            "output_nhwc": save_array(args.output_dir, "output_nhwc", output_nhwc.cpu().numpy()),
            "conv0_weight_krsc": save_array(args.output_dir, "conv0_weight_krsc", conv0_weight_krsc),
            "conv0_bias": save_array(args.output_dir, "conv0_bias", model.conv0.bias.detach().cpu().numpy()),
            "conv1_weight_krsc": save_array(args.output_dir, "conv1_weight_krsc", conv1_weight_krsc),
            "conv1_bias": save_array(args.output_dir, "conv1_bias", model.conv1.bias.detach().cpu().numpy()),
        },
    }

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")
    print(f"output_nhwc shape={tuple(output_nhwc.shape)}")


if __name__ == "__main__":
    main()
