# conv-fused

This directory contains a first correctness-oriented PyTorch CUDA extension for:

```text
conv1x1 -> relu -> conv1x1 -> bilinear
```

The exported function is:

```python
conv_fused.conv1x1_relu_conv1x1_bilinear(
    input, weight0, bias0, weight1, bias1, output_size,
    align_corners=False, scales_h=None, scales_w=None
)
```

Inputs use PyTorch's NCHW and OIHW layouts. The CUDA kernel computes the complete chain in one launch and matches:

```python
y = torch.nn.functional.conv2d(input, weight0, bias0)
y = torch.nn.functional.relu(y)
y = torch.nn.functional.conv2d(y, weight1, bias1)
y = torch.nn.functional.interpolate(y, size=output_size, mode="bilinear", align_corners=align_corners)
```

Build and check from the repository root:

```powershell
cmake -S . -B build -DCMAKE_PREFIX_PATH="$(python -c "import torch; print(torch.utils.cmake_prefix_path)")"
cmake --build build --target conv_fused_check --config Release
```

The current kernel prioritizes PyTorch parity. It keeps the code isolated so a later CUTLASS back-to-back 1x1 implementation, following `examples/13_two_tensor_op_fusion`, can replace the inner computation without changing the Python-facing contract.
