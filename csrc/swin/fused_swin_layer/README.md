# Fused Swin Layer

完整 DLSS4 block 的独立 CUTLASS 2.x 实现：两次 RMSNorm、attention、两次残差、
32→128→32 的 exact GELU MLP，在一个 kernel 内完成，无外部 workspace。
目标 SM80/SM89，输入 FP16、MMA/归约 FP32；C=32、G=1、Dq=Dv=32、window=4、ratio=4。
支持 shared/separate QK、shift_h/shift_w 0..3、reflect padding/crop 和可选 projection bias。

按 `cutlass/gemm/kernel/default_gemm.h` 的工厂方式组装：

| 本地层次 | 职责与实际使用的库组件 |
|---|---|
| [ops](ops/fused_swin_layer.h) | scalar Problem、raw device pointer Arguments |
| [device](device/fused_swin_layer.h) | `can_implement/get_workspace_size/initialize/run/operator()` |
| [kernel factory](kernel/default_fused_swin_layer.h) | 选择 Mma、Epilogue，组装 CutlassKernel |
| [threadblock factory](threadblock/default_mma.h) | 从库 DefaultMma 取得 thread map、SmemIterator、MmaPolicy |
| [MMA](threadblock/mma.h) | 库 MmaBase::SharedStorage、RegularTileIterator、warp MMA；同步单 stage |
| [epilogue](epilogue/default_epilogue.h) | DefaultEpilogueTensorOp 的组件，原生 PredicatedTileIterator 的 UseCUDAStore |
| [完整层编排](threadblock/fused_swin_layer.h) | 库 global/shared iterator，RMSNorm、softmax、GELU、残差和舍入 |

没有自研 arch、smem_layout、iterator 类或 inline PTX。global 输入/权重/偏置/索引读取
使用原生 predicated iterator，输入 partition 使用 GatherA；shared TensorOp layout
由库 MmaCore 产生。halo 的负 scatter 索引在调用输出 iterator 前过滤。
本地 core 函数沿用 CUTLASS 接口名；测试脚本和 fixture helper 属于 host harness。

调用方传入的输入/输出为连续 NHWC；权重分别为 `[64或96,32]`、`[32,32]`、
`[128,32]`、`[32,128]`，position bias 已展开为 `[16,16]`。
gather/scatter 表按窗口排列，每窗口 16 行；gather 必须落在输入 token 范围内，
scatter 为有效输出行号或 -1，所有有效输出恰好有一个 writer。表可使用现有
`window_attention/window_index.h` 在 host 生成，harness 会核对范围与唯一覆盖。
所有数据指针要求 16-byte 对齐，索引指针要求 int 对齐；输出不得与输入/权重/索引重叠。

`DefaultGemm` 本身只有一个矩阵乘积。完整 block 需要局部 threadblock 编排，原因、
地址空间限制和原生 iterator 的边界见 [draft](docs/00-design/draft.md) 与
[plan](docs/00-design/plan.md)。当前每个窗口由一个 warp/CTA 处理，所有中间结果留在 shared memory。

从仓库根运行：

```bat
scripts\kernels\fused_swin_layer\fused_swin_layer.bat
```

入口执行 source gate → build → verify → bench；verify 失败立即停止。
验证数据与所有产物在 `build/fused_swin_layer/`。指定 `CUTLASS_ARCH=80` 可以在对应设备
执行同一入口；本机是 SM89，因此单独的 SM80 编译使用下述命令：

```powershell
cmake -S csrc/swin/fused_swin_layer -B build/fused_swin_layer/compile-80 -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build/fused_swin_layer/compile-80 --config Release --target fused_swin_layer
```

本机验证结果见 [结果与取舍](docs/00-design/results.md)。SM80 已编译，尚无 SM80 实机结果。
