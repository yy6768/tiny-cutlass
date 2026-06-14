# FlashAttention 记忆

这个目录保存迁移到 tiny-cutlass 内部的 FlashAttention 学习实现。

## 当前结构

- `00-naive-attention` 是 baseline，实现仍会 materialize 完整 `P`。
- `01-online-softmax` 只改变 softmax 路径，还不是 IO-aware tiling。
- `02-tiled-online-attention` 是当前 IO-aware tiled kernel 步骤，来自对 CUTLASS
  flash-attention example 的本地改写。
- `epilogue/`、`gemm/`、`iterators/`、`transform/` 是 `02` 需要的支持头文件。
- `blogs/` 只放学习笔记。

## 修改规则

- 先 build，再 verify，再 benchmark。
- 性能结论前必须先用选定 reference 做 correctness check。
- 共享测试逻辑保留在 `csrc/tests/flash-attention/flash_attention_test.cpp`；
  编号 kernel `.cu` 文件通过 `flash_attention.h` 暴露 launch entry。
- `00/01/02` 和后续迁移变体都要注册到共享 `Kernel` 接口，让
  `flash_attention_test --kernel=all` 可以一起覆盖。
- C++ 作用域跟随 CUTLASS example 风格：这个学习工作区避免多层项目 namespace；
  文件局部 helper 使用匿名 namespace。
- 当前 reference contract 是 cuDNN SDPA。缺少 cuDNN headers、frontend 或 import
  library 时，CMake 必须提前失败；不要加 `HAS_CUDNN` 风格的 C++ fallback 分支。
- 尽量减少 fallback implementation。不支持的 arch、dtype 或 dependency 应显式拒绝，
  不要静默选择更弱的隐藏路径。
- 当前优化工作聚焦 SM80 FP16 和 SM89 FP8，除非用户明确扩大范围。
- profiling artifact 必须放在 `build/` 下。
- step 编号保持清晰，并且局限在这个 family 内。
- 不要把 attention 特定假设扩散到目录外，除非用户明确要求更大范围的重构。
