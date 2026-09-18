# 从资料到 kernel

先明确 operator/problem、GPU/CC、dtype/accumulator、layout/stride/alignment、shape 范围、数值容差、CUDA/CUTLASS 版本。已有本地实现时，先读它的 policy 与 launcher，减少无关检索。

1. 按架构索引找接近的实现。读取 PR 的 changed files 和固定 SHA，或官方示例；检查真正的 MMA/Copy atom、tile、pipeline、epilogue 和 dispatch 条件。标题或 GPU 型号只是发现入口。
2. 查同一条路径的 issue，找 ragged shape、scale indexing、split-K、swizzle、SMEM 超限、编译器版本和 Windows/MSVC 约束。issue 的现象、作者解释和维护者确认分别看待。
3. 提出一个可以验证的改动，例如减少 stages 使 SM86 grouped GEMM 合法启动，或把 blockwise scale 索引改成逻辑 tile 坐标。不要直接套用上游给另一种 shape 选的 tile。
4. 实现依照 `cutlass-kernel` 与目标目录规则，policy 通过模板表达；核心层与 PyTorch ownership 分离。保持 build → verify → bench；准确报告不支持的组合，不悄悄改为 SIMT。
5. 正确性覆盖实际需求中的边界：非整 tile、不同 stride、bias/activation、split-K、不同 scale group 和零尺寸（若 API 支持）。通过后再比较 latency；记录 GPU、软件版本、shape、warmup/repeat、精度与 kernel variant。

本库不提供未经编译的“保证可运行”代码片段。源码链接是供阅读和移植的证据，不代表用户工作区已安装同样版本。上游 benchmark 是候选方向；本地 NCU 结论必须来自已通过 reference parity 的实际 kernel。

常见入口：

- GEMM/quantization：[按技术](../queries/by-technique.md)，寻找 `mma-sync/fp8/nvfp4/block-scaling/epilogue`。
- attention：[按算子](../queries/by-kernel-type.md)，关注 online softmax、head dimension、mask 和 SMEM。
- convolution/fusion：[Ada 融合讨论](../sources/issues/cutlass-2189.md) 与官方 implementation map；不能由 GEMM EVT 推断所有 conv device API 都支持相同融合。
- 访存/性能异常：[按问题](../queries/by-problem.md)，结合具体 load/store、bank conflict 与寄存器数据。
