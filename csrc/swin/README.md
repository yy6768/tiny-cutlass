# Swin CUTLASS

这个目录只保留一条 Swin 主路径，不再按 FlashAttention 那样拆 00/01/02/03 学习变体。
目标是把 FasterTransformer 的 Swin `WindowAttention` 路径搬成一个可读、可验证的 CUTLASS 实现骨架。

## 结构

- `swin_cutlass.cu`: 单一测试入口，串起 Swin block 的 attention 主路径。
- `kernels/swin_kernels.h`: Swin layout glue kernel，包括 `WindowPartition`、`WindowReverse`、`PatchMerge`、QKV bias/split、output bias。

## 路径

```text
WindowPartition
QKV projection
WindowAttention
OutputProjection
WindowReverse
```

`WindowAttention` 内部使用 tiled online attention core，QK、softmax、PV 不作为公开接口暴露。

## FasterTransformer 对应表

| tiny-cutlass | FasterTransformer Swin |
| --- | --- |
| `WindowPartition` | `invokeShiftPartition` |
| `QKV projection` | `WindowAttention<T>::forward` 里的 `attention_qkv_kernel` GEMM |
| `WindowAttention` | `FusedMHARunnerFP16v2::run` / fused local attention |
| `OutputProjection` | `attention_output_weight.kernel` GEMM |
| `WindowReverse` | `invokeReverseRoll` |
| `PatchMerge` | stage transition 的 image merge gather |

## 当前约束

- dtype: FP16 input/weight/output，accumulation 使用 FP32。
- layout: `[B, H, W, C]` 输入，window token 为 `[B * num_windows, window_len, C]`。
- `C = head_number * head_size`。
- `image_size % window_size == 0`。
- `0 <= shift_size < window_size`。
- `head_size % 8 == 0`，`C % 8 == 0`。
- 当前 teaching path 只支持 `window_len <= 64`，覆盖 Swin-T 常见 `window_size=7`。

## 构建和验证

```bat
cmake -S . -B build -DTINY_CUTLASS_BUILD_FLASH_ATTENTION=OFF -DTINY_CUTLASS_BUILD_SWIN=ON -DTINY_CUTLASS_BUILD_CONV_FUSED=OFF
cmake --build build --config Release --target swin_cutlass_test
build\csrc\swin\Release\swin_cutlass_test.exe --batch_size=1 --image_size=14 --window_size=7 --shift_size=3 --head_number=3 --head_size=32 --iterations=1 --reference-check=true
```

脚本入口：

```bat
scripts\kernels\swin\swin.bat
```
