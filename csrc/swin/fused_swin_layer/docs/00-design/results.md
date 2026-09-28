# 验证结果与候选取舍

2026-09-15：RTX 4070 Laptop GPU（SM89）、CUDA 12.9.41、MSVC 19.44.35228、
PyTorch 2.7.1+cu126。完整 reference 复用 `csrc/tests/swin/block_reference.py`，不修改公式或容差。

| 证据 | 结果 |
|---|---|
| 本地源码门禁 | 9 个 core 头文件通过，函数接口有 CUTLASS 定义依据 |
| SM89 / SM80 编译 | 均成功；ptxas 78 registers，无 stack/spill |
| SM89 reference | 24/24 完整 block 用例通过，包括三组 720p 输入 |
| 误差上限 | 所有用例最大 MAE 为 9.53674e-7，最大 max_abs 为 0.00292969 |
| CUDA Graph | 所有用例为单 kernel 节点，清空输出后的 replay parity 通过 |
| 不支持配置 | 错误 shape、dropout、FP8、epsilon、空指针、对齐及别名等显式拒绝 |
| workspace / shared memory | 外部 workspace 0；每 CTA shared memory 19,456 bytes |
| Compute Sanitizer memcheck | singleton+reflect+graph 用例，0 errors |
| Compute Sanitizer racecheck | shift_33+graph 用例，0 errors / 0 warnings |
| SM80 实机 / NCU | 未运行；不能用 SM89 数据替代 |

性能口径为完整 block CUDA Graph，warmup 20 次，每次 50 iterations，重复 3 轮并交替
baseline/candidate 顺序；每个进程计时前重新通过同输入 parity。以下单位为 ms，取中位数。
baseline 是已有 `build/swin/tests/swin/Release/swin_block.exe`；两个二进制的 SHA256
写入 benchmark-comparison.json，不把旧二进制的来源默认为当前工作树重新编译结果。

| 输入 | 旧 block | 本候选 |
|---|---:|---:|
| 720×1280，shared QK，shift=0 | 3.55869 | 6.45323 |
| 720×1280，shared QK，shift=2 | 3.61329 | 6.51477 |
| 720×1280，separate QK，shift=2 | 3.81628 | 6.74628 |

本候选完成用户要求的库组件与结构重建，保留为独立工作区。当前耗时约为旧版的
1.77～1.81 倍，不作为性能替代版本晋升。同步单 stage、每窗口一个 warp，以及逐行
非线性处理是后续可测候选；尚未用 NCU 证明它们分别贡献了多少耗时。

证据目录相对仓库根：

- `build/fused_swin_layer/build.log`、`compile-80/build.log`
- `build/fused_swin_layer/verification.json`、`fixtures/*/verify.log`
- `build/fused_swin_layer/benchmark-comparison.json`、`benchmark-runs/20260914T173104774536Z/fixtures/*/{candidate,baseline}-*.log`
- `build/fused_swin_layer/memcheck.log`、`racecheck.log`

source/header SHA256 和 binary SHA256 记录在 verification.json；性能报告记录对应二进制。
这些产物按仓库约定在 build 下，不纳入源码提交。

一键 `.bat` 已实际完成 build → 24 用例 verify → 三组 720p bench，日志为
`build/fused_swin_layer/workflow.log`。每次 benchmark 另建带 UTC 时间戳的证据目录。
