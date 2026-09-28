# NATTEN 1D forward

这是从 NATTEN 的 1D non-causal 邻域规则出发的 CUTLASS 2.x TensorOp 实验。
实际计算在本工作区的 device → kernel → threadblock → warp → thread → epilogue
分层中完成，不再调用 example 41 的整体 `AttentionKernel`，也不预生成全局 mask。

## 执行路径

1. `device/nb_atten.h` 的 `NBAtten<Kernel>` 是设备算子。公开的
   `Arguments` 只包含连续 Q/K/V/O 指针、可空 LSE 指针、`NeighborhoodProblem`。
   `initialize(args)` 检查条件并构造 `Kernel::Params`；`run(stream)` 用
   `cutlass::Kernel<Kernel>` 启动；`operator()(args,stream)` 是一次性入口。
   `get_workspace_size(problem) == 0`。没有内部设备内存分配。
2. `kernel/default_nb_atten.h` 的 `DefaultNBAtten<ArchTag,Element,ThreadblockShape,
   WarpShapeQK,WarpShapePV,Layout>` 选择标准 CUTLASS ThreadMap、global/shared
   iterator、warp TensorOp、概率 fragment iterator、Mma 与 epilogue。
   它是类型工厂，不是运行时启动器。当前实例固定物理 64×64 tile、D/Dv=64；
   运行时 D/Dv 可为 8 的倍数且 ≤64，global iterator 将不足部分置零。
3. `kernel/nb_atten.h` 的 `NeighborhoodKernel` 定位 `(query_tile,head,batch)`，
   用窗口首尾求该 tile 的 K/V 并集，确定 KV tile 次数和尾块有效行数，
   构造初始 CUTLASS iterator、Mma 与 epilogue。
4. `threadblock/nb_atten_mma.h` 的 `NeighborhoodMma` 持有 Q/K/V shared storage。
   先用标准 iterator + `cp_async_zfill` 搬入 Q；每轮搬一块 K/V、同步、运行
   QK TensorOp → softmax → PV TensorOp，再推进 K/V iterator。
   完整主循环、shared 重用与同步在这里。
5. `warp/nb_atten_mma.h` 用 `DefaultMmaTensorOp` 的标准 fragment、iterator
   和 `mma.sync` 执行两个矩阵乘。`thread/nb_atten_softmax.h` 是 **attention
   自定义逻辑**：按每行窗口坐标遮挡 score，维护 online softmax 的 m/l，
   算出下一个输出的缩放系数。通用 CUTLASS GEMM 没有 neighborhood mask
   或 online softmax 语义。
6. `epilogue/nb_atten_epilogue.h` 用 CUTLASS fragment/output iterator 将
   已归一化的 FP32 O 转为 FP16，并按需写自然对数 LSE。

每个 CTA 处理 64 个 Q 行，两个 warp 各负责 32 行。K/V 的扫描范围是该
Q tile 各行窗口的并集；每行有效 key 由 thread 层精确判断。
输入 Q/K 为 `[B,L,H,D]`，V/O 为 `[B,L,H,Dv]`，LSE 为
`[B,H,round_up(L,32)]`，padding 不属于有效输出。

## 来源与职责依据

- NATTEN `92750c3cf837652d58b6091e5ebd37dcad46e753` 的
  [`NeighborhoodAttentionMaskBase<false>`](https://github.com/SHI-Labs/NATTEN/blob/92750c3cf837652d58b6091e5ebd37dcad46e753/csrc/include/natten/cuda/fna/na_utils.cuh)
  给出窗口边界与偶数窗口的语义；此处仅实现 stride=dilation=1 的简化式
  `include/natten/neighborhood.h`，保留 `LICENSE-NATTEN`。
- CUTLASS [example 13](../../3rdparty/cutlass/examples/13_two_tensor_op_fusion/threadblock/b2b_mma_multistage.h)
  的 `copy_tiles_and_advance_0` 作为 threadblock 内 iterator 搬运的职责参考，
  [DefaultB2bGemm](../../3rdparty/cutlass/examples/13_two_tensor_op_fusion/kernel/default_b2b_gemm.h)
  作为模板工厂边界参考。
- CUTLASS [example 35](../../3rdparty/cutlass/examples/35_gemm_softmax/gemm_with_softmax.h)
  区分 GEMM 与 softmax 专用计算；[example 41](../../3rdparty/cutlass/examples/41_fused_multi_head_attention/kernel_forward.h)
  提供 QK、online softmax、PV 和 LSE 的数值流程参考。当前没有复用或复制
  example 41 的私有 `Params`、MMA、Loader 或 epilogue 实现。
- [CUTLASS Gemm device](../../3rdparty/cutlass/include/cutlass/gemm/device/gemm.h)
  是 `Arguments`、`Params`、`initialize`、`run` 和 `operator()` 的接口参考；
  [cutlass::Kernel](../../3rdparty/cutlass/include/cutlass/device_kernel.h)
  是启动 trampoline。

## 启动契约

```cpp
using Factory = tiny_cutlass::natten::DefaultNBAtten<
    cutlass::arch::Sm80, cutlass::half_t,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 64, 16>,
    cutlass::gemm::GemmShape<32, 64, 16>>;
using Device = tiny_cutlass::natten::NBAtten<typename Factory::Kernel>;
Device::Arguments args{q, k, v, out, lse, problem};
Device op;
cutlass::Status status = op.initialize(args);
if (status == cutlass::Status::kSuccess) status = op.run(stream);
```

包含 `kernel/default_nb_atten.h` 与 `device/nb_atten.h`。
指针需 16-byte 对齐，LSE 可以为 null；输入输出不能重叠。
当前只支持 FP16、非 causal 1D、BLHD 连续布局、`stride=dilation=1`、
`1 <= kernel_size <= L`、D/Dv 为 8 的倍数且 ≤64。默认 CMake 编译 `sm_89`，
`Sm80` 类型表示 warp TensorOp 指令的最低架构。运行时只接受 SM80/SM89，
但 SM80 仍未做本地 parity。其他配置显式失败。

## 验证与计时

入口：`scripts/kernels/natten/natten.bat`。顺序为 build → verify → bench；
verify 失败后不计时，构建与报告在 `build/natten/`。
`csrc/tests/natten/fna.cu` 使用 FP16 输入和独立 double host reference，
同时验证 O/LSE、偶数窗口、首尾、非整 tile、不同 D/Dv、可空 LSE 与拒绝路径。
默认误差阈值为 O MAE ≤1e-3、最大误差 ≤1e-2、LSE 最大误差 ≤1e-3。
计时范围是单个 device kernel 调用，包含 host dispatch 空隙；不包含分配与
H2D/D2H。当前没有 NCU 分析或跨实现加速比结论。
