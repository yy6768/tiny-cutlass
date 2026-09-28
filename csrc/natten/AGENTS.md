# NATTEN 工作区

- 当前是独立 1D non-causal FNA forward，FP16、BLHD、stride=dilation=1。
- `device/nb_atten.h` 的 `NBAtten<Kernel>` 是设备算子，提供
  `Arguments`、`can_implement`、`initialize`、`run`、`operator()`。
  公开输入使用 raw pointers + `NeighborhoodProblem` + `cudaStream_t`；
  不引入 PyTorch、ATen、pybind 或内部 workspace 分配。
- `kernel/default_nb_atten.h` 是模板工厂，`kernel/nb_atten.h` 只负责 CTA
  定位、早退、KV 次数、初始 iterator 和 Mma/epilogue 组装。
- `threadblock/nb_atten_mma.h` 负责完整 KV 主循环、标准 CUTLASS iterator
  搬运、shared reuse 和同步；`warp/nb_atten_mma.h` 组合 CUTLASS TensorOp。
- `thread/nb_atten_softmax.h` 是 attention 专用的逐行 mask 和 online softmax；
  `epilogue/nb_atten_epilogue.h` 负责转换、O/LSE 写回。
- 不新增独立 Loader 层、自定义 MMA 指令或 SIMT attention fallback；
  参考 example 13 的 threadblock 搬运职责，不复制 example 41 私有实现。
- 新增 stride/dilation/causal/2D/3D 时先扩 problem、reference 和支持检查。
- 测试入口 `csrc/tests/natten/fna.cu`，target `natten_fna`，脚本
  `scripts/kernels/natten/natten.bat`，顺序 build → verify → bench。
- O 与 LSE 都必须对齐独立 reference；失败时不得 benchmark。
- 所有构建输出和报告只放 `build/`；保留 NATTEN MIT 来源与许可证。
