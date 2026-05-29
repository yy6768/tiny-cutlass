import argparse
import importlib.util
import pathlib
import sys

import torch
import torch.nn.functional as F


def load_module(module_dir: pathlib.Path):
    candidates = sorted(module_dir.glob("conv_fused*.pyd")) + sorted(module_dir.glob("conv_fused*.so"))
    if not candidates:
        raise FileNotFoundError(f"conv_fused extension was not found in {module_dir}")

    spec = importlib.util.spec_from_file_location("conv_fused", candidates[0])
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def reference(x, w0, b0, w1, b1, output_size, align_corners, scales):
    y = F.conv2d(x, w0, b0)
    y = F.relu(y)
    y = F.conv2d(y, w1, b1)
    if scales is None:
        return F.interpolate(y, size=output_size, mode="bilinear", align_corners=align_corners)
    return F.interpolate(y, scale_factor=scales, mode="bilinear", align_corners=align_corners)


def run_case(module, dtype, shape, hidden, out_channels, output_size, align_corners, use_bias, scales=None):
    torch.manual_seed(1234)
    device = torch.device("cuda")
    n, c, h, w = shape

    x = torch.randn(shape, device=device, dtype=dtype)
    w0 = torch.randn((hidden, c, 1, 1), device=device, dtype=dtype)
    w1 = torch.randn((out_channels, hidden, 1, 1), device=device, dtype=dtype)
    b0 = torch.randn((hidden,), device=device, dtype=dtype) if use_bias else None
    b1 = torch.randn((out_channels,), device=device, dtype=dtype) if use_bias else None

    scales_h = None
    scales_w = None
    fused_output_size = list(output_size)
    if scales is not None:
        scales_h, scales_w = scales
        fused_output_size = [int(h * scales_h), int(w * scales_w)]

    actual = module.conv1x1_relu_conv1x1_bilinear(
        x, w0, b0, w1, b1, fused_output_size, align_corners, scales_h, scales_w
    )
    expected = reference(x, w0, b0, w1, b1, output_size, align_corners, scales)
    torch.cuda.synchronize()

    if dtype is torch.float16:
        atol, rtol = 5e-2, 5e-2
    else:
        atol, rtol = 2e-4, 2e-4

    max_abs = (actual - expected).abs().max().item()
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    print(
        f"pass dtype={dtype} shape={shape} hidden={hidden} out={out_channels} "
        f"output={tuple(fused_output_size)} align_corners={align_corners} "
        f"bias={use_bias} max_abs={max_abs:.6g}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for conv_fused_check")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    module = load_module(args.module_dir)

    cases = [
        (torch.float32, (1, 3, 5, 7), 4, 2, (9, 11), False, True, None),
        (torch.float32, (2, 5, 6, 4), 7, 3, (3, 8), True, False, None),
        (torch.float32, (1, 4, 3, 5), 6, 4, (6, 10), False, True, (2.0, 2.0)),
    ]
    if torch.cuda.get_device_capability()[0] >= 7:
        cases.append((torch.float16, (1, 8, 4, 4), 8, 4, (7, 7), False, True, None))

    for case in cases:
        run_case(module, *case)


if __name__ == "__main__":
    sys.exit(main())
