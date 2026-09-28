# Fused Swin Layer

继承根与 `csrc/swin/AGENTS.md`，使用 `cutlass-kernel` 和 `kernel-design-agents`。
这是独立的完整 DLSS4 block 实现，语义和边界见 `docs/00-design/plan.md`。

- 工厂对照 `cutlass/gemm/kernel/default_gemm.h`，组装 `Mma`、`Epilogue` 和 `Kernel`。
- 仅维护 SM80/SM89，配置在模板参数和 CMake；FP16/FP32 基线。
- 本地 device 函数沿用 CUTLASS 接口名；不新增搬运/MMA 动词包装。
- global、shared、accumulator 访问均使用 CUTLASS iterator。
- 不新增 arch、smem_layout 或 bespoke iterator；必要融合编排写在 threadblock 层。
- 新候选先独立构建、完整 reference parity，再注册到父工作区 target。
- harness 在 `csrc/tests/fused_swin_layer/`，产物在 `build/fused_swin_layer/`。
