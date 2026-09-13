# 可执行计划

1. 在 ops/swin_block.h 定义 raw-pointer block 参数和入口；kernel/default_swin_block.h
   组装 CUTLASS policy，threadblock/swin_block_mma.h 实现 norm/attention/residual/MLP。
2. attention MMA 接受可选 CTA 内输出，保持原 attention API 行为；基础 MMA helper
   接受兼容的 shared operand storage，便于复用，不写矩阵乘法替代实现。
3. csrc/tests/swin/block.cu + block_reference.py 验证小图、全部 shift、norm 语义、
   separate QK，以及 full 720×1280/C32/ratio4。测试 graph 单节点与非默认 stream。
4. 00 通过后保存二进制与源码。实现 01，重跑相同契约，再交替 graph benchmark；
   选择稳定优于 00 的候选。失败候选及测量写 build/swin/block/。
5. verify.py/bench.py 增加 --block，swin.bat block 执行 build→verify→bench。
6. 对胜出版本进行 sanitizer、NCU source/CSV、NSYS，并生成新的 HTML 全尺寸报告。
   只将同一完整 block、同一输入和相同计时方式的测量用于性能结论。

已验证候选 02：在 01 的基础上使用 8 个 half / 128-bit 的 operand load/store，
两个 RMSNorm 改为 4-lane 一行、每 lane 8 channels。CUTLASS 的 crosswise layout
在 128-bit vector 内保持连续，依据 `tensor_op_multiplicand_sm75.h` 的
`coord.contiguous() % kElementsPerAccess` 映射。全尺寸结果见
`build/swin/block/bench-02/bench.json`；23 个完整 block 用例通过。

最终采用 02。00/02 五轮交替测量的中位数：无 shift 26.024→3.572 ms，
reflect shift=(2,2) 26.255→3.700 ms，separate QK + reflect 29.387→3.902 ms。
设备为 RTX 4070 Laptop / SM89；不是 4090 实测值。原始样本与 binary hash 在
`build/swin/block/final/bench.json`，可视化及复现材料在 `build/swin/block/index.html`。
23 个 block 用例、66 个 attention 回归、完整 shifted 720p 的 memcheck/synccheck
以及小图 racecheck 均通过。NCU full/source 内嵌源码、CSV 与 NSYS 已保留，
确认单 kernel、128-bit LDG/STS、HMMA.16816；`swin.bat block` 全流程实跑成功。
