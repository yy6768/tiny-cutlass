# 06 验证与 Profiling 工作流

## 统一入口

```bat
scripts\kernels\swin\run.bat
```

顺序固定为：

```text
build -> verify.py -> bench.py
```

`verify.py` 运行 PatchEmbed、SwinAttention 和 SwinBlock reference。官方 checkpoint
artifact 由 `checkpoint/manifest.json` 描述；工作流只消费已有 artifact，不在 build
或 verify 阶段下载模型。

## Benchmark

`bench.py` 是唯一性能入口。普通模式记录 runtime/CSV/JSON；设置
`CUTLASS_PROFILE=1` 后，`run.bat` 给同一个 bench 阶段增加 `--nsys --ncu`。

NCU 产物写入 `profile/swin/ncu/<case>/`：

```text
<case>.ncu-rep
<case>.csv
ncu_kernel_times.json
```

CSV 解析使用 kernel name 和原始 metric，不再维护一个按 launch 序号猜测 kernel 名的
独立报告脚本。reference parity 未通过时，`run.bat` 不会进入这个阶段。

## 当前结论

验证、benchmark 和 profiling 共享同一组 executable 与 case 定义。这样结构变化后，
不会出现独立 profiling 脚本继续描述旧 launch 序列的问题。
