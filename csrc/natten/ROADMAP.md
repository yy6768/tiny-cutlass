# NATTEN 实现路线图

这份路线图把当前接口骨架推进到可验证、可 benchmark、可和 Swin 对齐的 CUTLASS
FNA kernel。它不是性能结论；没有真实 kernel 和 reference parity 前，任何性能分析都
只能是瓶颈假设。

## 里程碑 0：当前状态

- 已保留 `DefaultFnaForwardPolicy`、`FnaCausalMask`、`FnaForwardProblem`。
- 已删除 fake runtime/status path。
- 已删除 custom MMA、custom iterator、custom epilogue、autogen symbol。
- `scripts\kernels\natten\natten.bat` 当前只做 build + 接口冒烟测试。

## 里程碑 1：reference parity

先补可审计 reference，再写真实 kernel。

最小 correctness matrix：

| 维度 | 初始覆盖 |
| --- | --- |
| rank | 1D |
| mask | non-causal，后续补 causal |
| dtype | FP16 input/output，FP32 accumulation |
| shape | `B in {1, 2}`，`L in {8, 16, 64}`，`heads in {1, 2}` |
| head dim | `head_dim in {16, 32}`，覆盖 `head_dim != head_dim_value` |
| neighborhood | `kernel_size in {3, 5, 33}` |
| dilation | `dilation in {1, 2}` |
| 输出 | `O`，后续如果 kernel 产出则校验 `logsumexp` |

reference 优先级：

1. 小 shape host reference，便于审计边界、dilation、mask。
2. PyTorch/NATTEN 官方实现，用于中等 shape parity。
3. benchmark 只在 reference parity 通过后启用。

## 里程碑 2：CUTLASS 原生 device 层

真实实现必须保持模板化：

- `DefaultFnaForwardPolicy<Rank, CausalMask, Element, ArchTag, ThreadblockShape, ...>`
  继续是主 policy 工厂。
- device/launcher 可以新增，但名字不能写死架构。
- arch、dtype、tile、mask 都通过模板参数或显式 problem/policy 表达。
- 底层优先复用 `3rdparty/cutlass` 的 TensorOp/FMHA/CuTe 能力。
- 不支持的 arch/dtype/shape 显式拒绝；不要退回 SIMT/raw CUDA fallback。

## 里程碑 3：benchmark 和 profiling

`.bat` 需要保持 build -> verify -> bench：

1. build：构建 NATTEN test/benchmark target。
2. verify：先跑 reference parity；失败则停止。
3. bench：只跑已通过 correctness 的 shape。
4. profile：Nsight artifact 输出到 `build/reports/natten`。

建议 benchmark matrix：

| 场景 | shape |
| --- | --- |
| 小型调试 | `B=1, L=64, heads=1, head_dim=32, kernel_size=33` |
| NAT 常见 1D 压力 | `B=2, L=1024, heads=8, head_dim=32, kernel_size=129` |
| dilation 压力 | `B=2, L=1024, heads=8, head_dim=32, kernel_size=65, dilation=2` |
| V 维度不等 | `head_dim=32, head_dim_value=64` |

每条结果必须记录：

- GPU、CUDA、CMake arch、dtype、shape。
- reference 名称和误差阈值。
- kernel time、effective bandwidth、可解释的 FLOP/byte 估计。
- Nsight Compute/Systems artifact 路径。

## 需要验证的瓶颈假设

| 假设 | 需要看的证据 |
| --- | --- |
| 非 fused 路径会被中间 score/probability HBM 写回拖慢 | DRAM throughput、写流量、是否 materialize `P` |
| sliding neighborhood 造成重复 K/V load | L2 hit rate、global load transaction、shared memory reuse |
| 边界/dilation/mask predicate 造成无效 work | branch efficiency、warp stall reason、active mask |
| tile shape 与 `kernel_size/head_dim` 不匹配导致 Tensor Core 利用率低 | tensor pipe utilization、eligible warps、sm throughput |
| softmax/logsumexp 占用寄存器和 shared memory | register count、occupancy、shared memory usage |

## 和 Swin 工作区统一的方向

可以统一：

- README 结构：范围、接口、reference、验证和 benchmark。
- 脚本结构：一个 family 一个 `.bat`。
- 测试位置：`csrc/tests/<family>`。
- CMake target 命名和 build output。
- benchmark 结果必须绑定 reference parity。

不能强行统一：

- Swin 是固定窗口 dense attention；NATTEN 是滑动邻域 attention。
- Swin 的 mask 更像 dense bias；NATTEN 的 mask/边界来自 per-query neighborhood。
- Swin 当前有完整 path + cuDNN reference；NATTEN 目前只有接口骨架。

后续重构时，应该统一工程组织，不要把 NAT/FNA 语义改成 Swin window attention。
