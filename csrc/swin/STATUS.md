# Swin Status

## 2026-07-14

- public API 收敛为 `PatchEmbed<ArchTag, Element>`、
  `SwinAttention<ArchTag, Element>`、`SwinBlock<ArchTag, Element>`。
- 删除 public `<Policy>` 入口；`DefaultPatchEmbed`、`DefaultSwinAttention`、
  `DefaultSwinBlock` 只在 CUDA 实现侧选择。
- 拆分 `SwinAttentionProblem/Tensors` 与 `SwinBlockProblem/Tensors`；共享部分收敛到
  `WindowAttentionTensors`。
- 从 SwinAttention 中移除 optional PatchMerge side path。PatchMerging 当前未实现为
  public operator。
- cuDNN reference 文件改为 `csrc/tests/swin/reference.h/.cpp`，注释明确它只用于
  correctness reference。
- workflow 入口改为 `scripts/kernels/swin/run.bat`；删除运行时 checkpoint 下载脚本，
  Nsys/NCU 采集与 CSV 解析合并进 `bench.py`。
- 文档整理为 `docs/00` 到 `docs/06`，旧 Policy、旧脚本名和过期实现说明已移除。

## 当前验证

- `scripts/kernels/swin/run.bat` 在 VS 2022 / CUDA 12.9 / RTX 4070 Laptop SM89
  环境完整通过 fresh build -> verify -> bench。
- verify 覆盖 PatchEmbed、两组 SwinAttention smoke、三组 SwinBlock host reference，
  并覆盖 `checkpoint/manifest.json` 的全部 official PatchEmbed/SwinAttention cases。
- SwinBlock 三组 MAE 分别约为 `5.85e-6`、`4.91e-6`、`5.07e-6`，低于 `3e-2`
  tolerance。
- `bench.py --case patch_embed --ncu --ncu-set basic` 已生成 `.ncu-rep`、raw CSV 和
  `ncu_kernel_times.json`；JSON 成功解析 ImagePad、filter reorder、Conv2d、LayerNorm
  四个 kernel 的 duration、throughput、grid/block、register 和 shared-memory metric。
