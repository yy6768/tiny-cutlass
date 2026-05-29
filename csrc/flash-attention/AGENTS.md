# FlashAttention 学习
需要你符合CUTLASS规范, 帮我学习FlashAttention，
我边写代码， 边写博客@blogs

## 当前状态
我现在正在模仿 examples 41 的风格， 写一个完全没有优化的naive attention但是当前未跑通。

## DO and NOT DO
- 符合 “CUTLASS style” cuda风格，而不是滥用“Raw CUDA”风格。cutlass封装了大量好用的工具函数和代码。
- 一切以benchmark为准。而不是乱猜
- 这个目录下的修改要尽量局部、最小化，优先在现有 `kernel_forward.h` 模板里做参数化调整，不要为了抽象而抽象。
- 如果 MM0 / MM1 只是在 `layout`、`epilogue`、`外层 tile 遍历方向` 上有差异，优先继续复用同一套 GEMM 骨架，通过模板参数或局部条件分支来区分。
- 不要主动引入新的 `AttentionGemmKernel<Config>`、`MM0Config`、`MM1Config` 这类重构性封装，除非用户明确要求重构。
- 不要把本地学习笔记写成“架构设计重构说明”；这里的目标是理解 CUTLASS / FlashAttention 的实现路径，而不是重新设计一套抽象层。
- 写博客时优先对应现有代码路径、函数名、模板参数和数据流，不要先发散到理论上更漂亮的结构。
- 涉及性能判断时，先看实现和 benchmark，再下结论。
