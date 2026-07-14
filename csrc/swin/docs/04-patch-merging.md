# 04 PatchMerging

## 前言

PatchMerging 是 BasicLayer 之间的 stage transition，不是 WindowAttention 的一部分。
旧实现只做 `2x2` gather，并通过 `SwinAttention` 的 nullable `patch_merged` 指针触发，
这会让一个 operator 拥有两套不相干的输出语义，因此已经移除。

## 官方语义

```text
[B, H, W, C]
  -> 2x2 gather / concat
[B, H/2, W/2, 4C]
  -> LayerNorm
  -> Linear(4C, 2C)
[B, H/2, W/2, 2C]
```

## 下一步

如果恢复，应该新增独立的 `PatchMergingProblem/Tensors`、
`DefaultPatchMerging<ArchTag, Element, ...>` 和 `device::PatchMerging<ArchTag, Element>`，
并对齐完整 gather + LayerNorm + reduction，而不是恢复旧 nullable side path。
