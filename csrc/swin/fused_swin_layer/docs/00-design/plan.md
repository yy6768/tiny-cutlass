# 完整层实现计划

目标 SM89（本机 RTX 4070 Laptop），另编译 SM80；SM80 实机验证单独记录。

1. `ops/` 定义 scalar Problem、raw pointer Arguments；不持有 Tensor。
2. `threadblock/` 用 `DefaultMma` 的 iterator/policy 组合共享内存内的连续 GEMM，
   归一化、softmax、残差和舍入操作在 iterator fragment 上执行。
3. `kernel/default_fused_swin_layer.h` 只选 Mma、库 Epilogue、Kernel；
   `kernel/fused_swin_layer.h` 只负责一窗口一 CTA 的调度。
4. `device/` 提供 `can_implement/get_workspace_size/initialize/run/operator()`，
   显式检查 arch、shape、dtype、指针及资源；单 kernel，无外部 workspace。
5. `csrc/tests/fused_swin_layer/` 复用独立 block reference，覆盖现有 23 个用例、无输出 bias、
   graph 单节点 replay、错误配置、异步 stream。每个可执行程序在计时前再做 parity。
6. 独立 CMake 和 `.bat`：source gate → build → verify → bench；失败立即停止。
   完整 parity 后仍保留独立入口，是否替代旧 kernel 由性能证据决定。
7. 留下源码依据、编译、reference、性能状态与实际命令；源码通过不能代替运行证据。

函数名约束覆盖新 core：成员接口采用 CUTLASS 的 `operator()`、`can_implement`、
`get_workspace_size`、`initialize` 和 `run`；构造函数与类型名对应。
测试/reference/构建脚本的 host 函数不作为新增 kernel API。
