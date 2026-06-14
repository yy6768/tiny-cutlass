from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper
from torch import nn

import modelopt.torch.quantization as mtq


FP8_MAX = 448.0


class Conv1x1ReluConv1x1ReluReference(nn.Module):
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
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv1 = nn.Conv2d(
            hidden_channels,
            output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    def forward(self, input_nhwc: torch.Tensor) -> torch.Tensor:
        x = input_nhwc.permute(0, 3, 1, 2).contiguous()
        x = torch.relu(self.conv0(x))
        x = torch.relu(self.conv1(x))
        return x.permute(0, 2, 3, 1).contiguous()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("build/conv-fused/fp8-reference"))
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--output-channels", type=int, default=16)
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def save_array(output_dir: Path, name: str, array: np.ndarray) -> dict[str, Any]:
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


def save_scalar(output_dir: Path, name: str, value: float) -> dict[str, Any]:
    array = np.asarray([value], dtype=np.float32)
    return save_array(output_dir, name, array)


def quantizer_amax(module: nn.Module, quantizer_name: str) -> float:
    quantizer = getattr(module, quantizer_name)
    amax = getattr(quantizer, "amax", None)
    if amax is None:
        raise RuntimeError(f"{module} has no calibrated {quantizer_name}.amax")
    value = float(amax.detach().float().max().cpu().item())
    return max(value, 1.0e-6)


def scale_inv_from_amax(amax: float) -> float:
    return max(amax / FP8_MAX, 1.0e-12)


def to_e4m3_bytes(tensor: torch.Tensor, scale_inv: float) -> np.ndarray:
    scaled = (tensor.detach().float() / scale_inv).clamp(-FP8_MAX, FP8_MAX)
    encoded = scaled.to(torch.float8_e4m3fn)
    return np.ascontiguousarray(encoded.view(torch.uint8).cpu().numpy())


def add_qdq(
    nodes: list[onnx.NodeProto],
    value_infos: list[onnx.ValueInfoProto],
    tensor_name: str,
    scale_name: str,
    q_name: str,
    dq_name: str,
    shape: list[int],
) -> str:
    nodes.append(
        helper.make_node(
            "TRT_FP8QuantizeLinear",
            [tensor_name, scale_name],
            [q_name],
            domain="trt",
        )
    )
    nodes.append(
        helper.make_node(
            "TRT_FP8DequantizeLinear",
            [q_name, scale_name],
            [dq_name],
            domain="trt",
        )
    )
    value_infos.append(helper.make_tensor_value_info(q_name, TensorProto.UINT8, shape))
    value_infos.append(helper.make_tensor_value_info(dq_name, TensorProto.FLOAT16, shape))
    return dq_name


def fp16_initializer(name: str, array: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.ascontiguousarray(array.astype(np.float16)), name)


def build_modelopt_fp8_onnx(
    path: Path,
    input_shape: list[int],
    hidden_channels: int,
    output_channels: int,
    weight0_nchw: np.ndarray,
    weight1_nchw: np.ndarray,
    input_scale_inv: float,
    weight0_scale_inv: float,
    stage0_scale_inv: float,
    weight1_scale_inv: float,
    opset: int,
) -> None:
    batch, height, width, channels = input_shape
    input_nchw_shape = [batch, channels, height, width]
    stage0_shape = [batch, hidden_channels, height, width]
    output_nchw_shape = [batch, output_channels, height, width]
    output_nhwc_shape = [batch, height, width, output_channels]
    weight0_shape = [hidden_channels, channels, 1, 1]
    weight1_shape = [output_channels, hidden_channels, 1, 1]

    nodes: list[onnx.NodeProto] = []
    value_infos: list[onnx.ValueInfoProto] = []
    initializers = [
        fp16_initializer("conv0_weight", weight0_nchw),
        fp16_initializer("conv1_weight", weight1_nchw),
        fp16_initializer("input_scale_inv", np.asarray([input_scale_inv])),
        fp16_initializer("weight0_scale_inv", np.asarray([weight0_scale_inv])),
        fp16_initializer("stage0_scale_inv", np.asarray([stage0_scale_inv])),
        fp16_initializer("weight1_scale_inv", np.asarray([weight1_scale_inv])),
    ]

    nodes.append(
        helper.make_node(
            "Transpose",
            ["input_nhwc"],
            ["input_nchw"],
            perm=[0, 3, 1, 2],
        )
    )
    value_infos.append(
        helper.make_tensor_value_info("input_nchw", TensorProto.FLOAT16, input_nchw_shape)
    )

    input_dq = add_qdq(
        nodes,
        value_infos,
        "input_nchw",
        "input_scale_inv",
        "input_q",
        "input_dq",
        input_nchw_shape,
    )
    weight0_dq = add_qdq(
        nodes,
        value_infos,
        "conv0_weight",
        "weight0_scale_inv",
        "weight0_q",
        "weight0_dq",
        weight0_shape,
    )
    nodes.append(
        helper.make_node(
            "Conv",
            [input_dq, weight0_dq],
            ["conv0"],
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
        )
    )
    nodes.append(helper.make_node("Relu", ["conv0"], ["relu0"]))
    value_infos.append(helper.make_tensor_value_info("conv0", TensorProto.FLOAT16, stage0_shape))
    value_infos.append(helper.make_tensor_value_info("relu0", TensorProto.FLOAT16, stage0_shape))

    stage0_dq = add_qdq(
        nodes,
        value_infos,
        "relu0",
        "stage0_scale_inv",
        "stage0_q",
        "stage0_dq",
        stage0_shape,
    )
    weight1_dq = add_qdq(
        nodes,
        value_infos,
        "conv1_weight",
        "weight1_scale_inv",
        "weight1_q",
        "weight1_dq",
        weight1_shape,
    )
    nodes.append(
        helper.make_node(
            "Conv",
            [stage0_dq, weight1_dq],
            ["conv1"],
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
        )
    )
    nodes.append(helper.make_node("Relu", ["conv1"], ["relu1"]))
    nodes.append(
        helper.make_node(
            "Transpose",
            ["relu1"],
            ["output_nhwc"],
            perm=[0, 2, 3, 1],
        )
    )
    value_infos.append(helper.make_tensor_value_info("conv1", TensorProto.FLOAT16, output_nchw_shape))
    value_infos.append(helper.make_tensor_value_info("relu1", TensorProto.FLOAT16, output_nchw_shape))

    graph = helper.make_graph(
        nodes,
        "conv1x1_relu_conv1x1_relu_modelopt_fp8",
        [helper.make_tensor_value_info("input_nhwc", TensorProto.FLOAT16, input_shape)],
        [helper.make_tensor_value_info("output_nhwc", TensorProto.FLOAT16, output_nhwc_shape)],
        initializer=initializers,
        value_info=value_infos,
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_operatorsetid("", opset),
            helper.make_operatorsetid("trt", 1),
        ],
    )
    onnx.save(model, path)


def main() -> None:
    args = parse_args()
    if args.channels % 16 or args.hidden_channels % 16 or args.output_channels % 16:
        raise SystemExit("channels, hidden-channels, and output-channels must be multiples of 16")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("ModelOpt FP8 export requires CUDA for this workflow")

    dtype = torch.float16
    model = Conv1x1ReluConv1x1ReluReference(
        args.channels,
        args.hidden_channels,
        args.output_channels,
    ).to(device=device, dtype=dtype)
    model.eval()

    inputs = [
        torch.randn(
            args.batch,
            args.height,
            args.width,
            args.channels,
            device=device,
            dtype=dtype,
        )
        for _ in range(args.calibration_batches)
    ]

    def forward_loop(module: nn.Module) -> None:
        with torch.no_grad():
            for value in inputs:
                module(value)

    quantized_model = mtq.quantize(
        model,
        copy.deepcopy(mtq.FP8_DEFAULT_CFG),
        forward_loop=forward_loop,
    )
    quantized_model.eval()

    input_nhwc = inputs[0]
    with torch.no_grad():
        output_nhwc = quantized_model(input_nhwc)

    input_scale_inv = scale_inv_from_amax(quantizer_amax(quantized_model.conv0, "input_quantizer"))
    weight0_scale_inv = scale_inv_from_amax(quantizer_amax(quantized_model.conv0, "weight_quantizer"))
    stage0_scale_inv = scale_inv_from_amax(quantizer_amax(quantized_model.conv1, "input_quantizer"))
    weight1_scale_inv = scale_inv_from_amax(quantizer_amax(quantized_model.conv1, "weight_quantizer"))
    output_amax = max(float(output_nhwc.detach().float().abs().max().cpu().item()), 1.0e-6)
    output_scale_inv = scale_inv_from_amax(output_amax)

    stage0_alpha = input_scale_inv * weight0_scale_inv / stage0_scale_inv
    output_alpha = stage0_scale_inv * weight1_scale_inv / output_scale_inv

    onnx_path = args.output_dir / "conv1x1_relu_conv1x1_relu_reference.onnx"
    build_modelopt_fp8_onnx(
        onnx_path,
        [args.batch, args.height, args.width, args.channels],
        args.hidden_channels,
        args.output_channels,
        quantized_model.conv0.weight.detach().cpu().numpy(),
        quantized_model.conv1.weight.detach().cpu().numpy(),
        input_scale_inv,
        weight0_scale_inv,
        stage0_scale_inv,
        weight1_scale_inv,
        args.opset,
    )

    weight0_krsc = quantized_model.conv0.weight.detach().permute(0, 2, 3, 1).contiguous()
    weight1_krsc = quantized_model.conv1.weight.detach().permute(0, 2, 3, 1).contiguous()

    input_e4m3 = to_e4m3_bytes(input_nhwc, input_scale_inv)
    weight0_e4m3 = to_e4m3_bytes(weight0_krsc, weight0_scale_inv)
    weight1_e4m3 = to_e4m3_bytes(weight1_krsc, weight1_scale_inv)
    bias1_e4m3 = np.zeros((args.output_channels,), dtype=np.uint8)
    stage0_scale = np.full((args.hidden_channels,), stage0_alpha, dtype=np.float32)
    bias0 = np.zeros((args.hidden_channels,), dtype=np.float32)

    artifacts: dict[str, Any] = {
        "model": onnx_path.name,
        "operator": "conv1x1_relu_conv1x1_relu",
        "quantization": "modelopt FP8_DEFAULT_CFG e4m3 per-tensor",
        "layout": {
            "input": "NHWC",
            "output": "NHWC",
            "weights": "KRSC",
        },
        "seed": args.seed,
        "device": str(device),
        "dtype": "float16",
        "problem": {
            "input_nhwc": [args.batch, args.height, args.width, args.channels],
            "hidden_channels": args.hidden_channels,
            "output_channels": args.output_channels,
            "output_nhwc": list(output_nhwc.shape),
        },
        "scales": {
            "input_scale_inv": input_scale_inv,
            "weight0_scale_inv": weight0_scale_inv,
            "stage0_scale_inv": stage0_scale_inv,
            "weight1_scale_inv": weight1_scale_inv,
            "output_scale_inv": output_scale_inv,
            "stage0_alpha": stage0_alpha,
            "output_alpha": output_alpha,
        },
        "artifacts": {
            "input_nhwc": save_array(args.output_dir, "input_nhwc", input_nhwc.cpu().numpy()),
            "output_nhwc": save_array(args.output_dir, "output_nhwc", output_nhwc.cpu().numpy()),
            "input_e4m3": save_array(args.output_dir, "input_e4m3", input_e4m3),
            "conv0_weight_e4m3_krsc": save_array(args.output_dir, "conv0_weight_e4m3_krsc", weight0_e4m3),
            "conv1_weight_e4m3_krsc": save_array(args.output_dir, "conv1_weight_e4m3_krsc", weight1_e4m3),
            "stage0_scale": save_array(args.output_dir, "stage0_scale", stage0_scale),
            "bias0": save_array(args.output_dir, "bias0", bias0),
            "bias1_e4m3": save_array(args.output_dir, "bias1_e4m3", bias1_e4m3),
            "output_scale_inv": save_scalar(args.output_dir, "output_scale_inv", output_scale_inv),
            "output_alpha": save_scalar(args.output_dir, "output_alpha", output_alpha),
        },
    }

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")
    print(f"onnx={onnx_path}")
    print(f"output_nhwc shape={tuple(output_nhwc.shape)}")
    print(f"stage0_alpha={stage0_alpha:.8g} output_alpha={output_alpha:.8g}")


if __name__ == "__main__":
    main()
