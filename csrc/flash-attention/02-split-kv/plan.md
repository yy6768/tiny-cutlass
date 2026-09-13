# 02 split-KV 实现与验证计划

用户要求：沿博客的 LeetCUDA MMA split-KV 路线实现 CUTLASS 风格的
device / kernel / threadblock / warp / epilogue，随后验证并用 NCU 解释实现代价。

源起点：LeetCUDA `0983c6554c39a18c1ae53739030090d0db127338` 的
`kernels/flash-attn/mma/basic/flash_attn_mma_split_kv.cu`。对照算法和 warp 切法，
本实现使用 CUTLASS 4.5.2 的 stock iterators、warp MMA 和 trampoline，
不移植上游的 PTX 宏或 Torch ownership。安装机器为 SM89，SM80 仅为 policy 目标。

## 契约

- FP16 输入/输出，QK、PV 累加和 online softmax 统计为 FP32。
- 非因果、无 dropout/bias，连续 `[B,S,H,D]`，允许不同 Sq/Sk 与 sequence tail。
- 显式实例 D=Dv 为 32/64/96/128；其他 dtype/layout/head shape 明确拒绝。
- CTA score tile 64x64，8 warps 按 Q方向2、K方向4 切分；每 warp score tile32x16。
- CUTLASS congruous B 的窄 N tile 地址置换不满足这里的四 warp 偏移需求：
  PV 每 warp 使用32x32，内部 V 通道补齐到128；D=32/64/96有额外 PV 计算。
  Q/K shared pitch 补齐到64的倍数，QK仍只计算实际D。
- 不分配全局 S/P 或跨 CTA split workspace，P tile 在 shared memory 中转。
- global→shared 使用 CUTLASS `PredicatedTileAccessIterator`、
  `RegularTileAccessIterator` 与 `arch::cp_async_zfill`。device policy 选择2-stage：
  Q/V 单 buffer，K 双 buffer，并让 V copy、下一 tile K copy 与当前 QK/softmax 重叠。
- cuDNN SDPA parity：MAE<=1e-3，max abs<=1e-2，所有输出 finite。

## 具体实现

1. DefaultFlashAttn 模板工厂组装 layout、stock global/shared iterators、两个 warp MMA、
   mainloop 和 epilogue；保留简单全局类型，具体形状只在 launcher 实例化。
2. warp 类解释 CUTLASS C fragment 的坐标，调用 IteratorA/B、transform 与 MmaTensorOp。
3. threadblock pipeline：Q prologue、KV tile load、QK、跨4warps max/sum、online
   rescale、P shared store、PV；明确每个 barrier 保护的 shared-memory 生命周期。
4. epilogue 类独立 normalize/store；device 类执行参数、设备、对齐与资源检查并 launch。
5. 接到现有共享 Kernel 注册，launch TU 移至 device/flash_attn.cu，更新 CMake。

## 验证与证据

构建目录 `build/split-kv-study/cmake`；脚本入口
`scripts/kernels/attention/02-split-kv-attention.bat`，验证/计时驱动在
`csrc/tests/flash-attention/verify.py`、`bench.py`。
验证 D、B/H stride、Sq/Sk tail、不同 seed/输入幅度、非默认 stream、非法配置拒绝；
共享 --kernel=all 对00/01/02做回归。build→verify→bench，任何验证失败都停止。

在通过 cuDNN 的相同 shape 上用 NCU 采 full/source（若可用），保留 binary/source hash、
parity log、native report、CSV/HTML。当前已有的02-tiled报告属于废弃实现，不能复用。
博客沿现有 Overview 更新源码阅读顺序和实測 Profile，保留用户前言和人工后记位置。

## 本次进展

- 五层 CUTLASS 实现和共享注册已接通，独立与共享 executable 均已构建。
- 22组正向命令（30次cuDNN对照）和2组非法配置拒绝已通过。
- 测试输出使用同一个非默认stream上的NaN预填，防止漏写或跨stream初始化竞态。
- 最终二进制已重新验证、benchmark 和采集 NCU full/source；verification、NCU 与
  当前 executable 的 SHA256 一致，报告内嵌的执行源码也与当前文件一致。
- memcheck、racecheck、synccheck 在 D=96、多head与双sequence tail用例上均通过。
- 博客已补齐五层源码阅读与实测 Profile，后记保留给用户；具体数值和报告链接以博客为准。
