# 05 BasicLayer 与整网

## 前言

当前 workspace 到完整 `SwinBlock` 为止，还没有实现 BasicLayer 或整网 runtime。
这一点需要和“单个 block 已经完整”区分开。

## BasicLayer

一个 stage 通常由多个 block 组成，shift 在 `0` 和 `window_size / 2` 之间交替；除最后
一个 stage 外，尾部还要执行 PatchMerging。

```text
SwinBlock(shift=0)
SwinBlock(shift=window/2)
...
PatchMerging
```

## 整网

```text
PatchEmbed
  -> BasicLayer x 4
  -> final LayerNorm
  -> global average pool
  -> classifier
```

## 当前结论

现在不能把 TensorRT `SwinAttention` plugin 或单个 `SwinBlock` benchmark 当作整网性能。
下一阶段首先需要独立 PatchMerging，然后才有足够稳定的边界组装 BasicLayer。
