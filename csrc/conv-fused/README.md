# conv-fused

This directory contains correctness-oriented PyTorch CUDA extension kernels for:

```text
conv1x1 -> relu -> conv1x1 -> bilinear
conv1x1 -> relu -> conv1x1 -> avg_pool2d
conv1x1 -> relu -> conv1x1 -> max_pool2d
```

The exported function is:

```python
conv_fused.conv1x1_relu_conv1x1_bilinear(
    input, weight0, bias0, weight1, bias1, output_size,
    align_corners=False, scales_h=None, scales_w=None
)

conv_fused.conv1x1_relu_conv1x1_avg_pool2d(
    input, weight0, bias0, weight1, bias1,
    kernel_size, stride=[], padding=[0, 0],
    ceil_mode=False, count_include_pad=True, divisor_override=None
)

conv_fused.conv1x1_relu_conv1x1_max_pool2d(
    input, weight0, bias0, weight1, bias1,
    kernel_size, stride=[], padding=[0, 0], dilation=[1, 1],
    ceil_mode=False
)
```

Inputs use PyTorch's NCHW and OIHW layouts. The CUDA kernel computes the complete chain in one launch and matches:

```python
y = torch.nn.functional.conv2d(input, weight0, bias0)
y = torch.nn.functional.relu(y)
y = torch.nn.functional.conv2d(y, weight1, bias1)
y = torch.nn.functional.interpolate(y, size=output_size, mode="bilinear", align_corners=align_corners)
y = torch.nn.functional.avg_pool2d(y, ...)
y = torch.nn.functional.max_pool2d(y, ...)
```

Build and check from the repository root:

```powershell
cmake -S . -B build -DCMAKE_PREFIX_PATH="$(python -c "import torch; print(torch.utils.cmake_prefix_path)")"
cmake --build build --target conv_fused_check --config Release
```

The current kernel prioritizes PyTorch parity. It keeps the code isolated so a later CUTLASS back-to-back 1x1 implementation, following `examples/13_two_tensor_op_fusion`, can replace the inner computation without changing the Python-facing contract.

The CUTLASS two-convolution fusion path is block/register resident for the two convolution mainloops. Spatial tails such as bilinear interpolation and pooling introduce cross-pixel dependencies, so this first version computes each output element directly and recomputes the conv chain for the required source pixels/window. That is intentionally redundant but gives a small, testable API surface before replacing the inner conv chain with a CUTLASS resident implementation.
