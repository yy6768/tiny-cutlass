# 任务契约与源码依据

用户要求独立工作区、完整融合、按 `default_gemm.h` 组装，以及复用 CUTLASS 函数接口、
iterator、arch 和 shared layout。原 `fused_swin_layer` 为空；现有完整 reference 位于
`csrc/tests/swin/block_reference.py`，现有实现位于 `window_attention`。

语义：reflect partition → RMSNorm1 → QKV → `(QKᵀ+bias)*scale` → softmax → PV →
output projection+bias → residual1 → RMSNorm2 → fc1+bias → exact GELU →
fc2+bias → residual2 → reverse/crop。保留 eager FP16 各物化节点的舍入。

边界沿用现有完整 block：C=32、ratio=4、G=1、Dq=Dv=32、window=4；支持 shared/separate
QK、两个轴独立 shift 0..3、reflect 小图与尾部。FP8、dropout、其他 shape 明确拒绝。
输入/权重 FP16、MMA/reduction FP32；MAE≤1e-3，max_abs≤2e-2，非有限值失败。
主输入 `[1,720,1280,32]`；基线是原 block 和独立 PyTorch reference。

候选问题：用库的 tile/warp iterator 覆盖全部张量搬运，是否能保持完整单 kernel parity？
首轮以结构和正确性为目标，性能改善没有预设保证。

固定版本源码依据（相对仓库根）：

| 职责 | 库源码 |
|---|---|
| 工厂组装 | `3rdparty/cutlass/include/cutlass/gemm/kernel/default_gemm.h` |
| global iterator、MmaCore、SmemIterator、warp policy | `.../gemm/threadblock/default_mma.h` |
| shared storage 与 TensorOp layout | `.../gemm/threadblock/mma_base.h` |
| regular shared iterator | `.../transform/threadblock/regular_tile_iterator_pitch_linear.h` |
| gather 输入 | `.../transform/threadblock/predicated_tile_iterator.h` |
| epilogue 组装 | `.../epilogue/threadblock/default_epilogue_tensor_op.h` |
| shared 输出 | `.../epilogue/threadblock/predicated_tile_iterator.h` 的 `UseCUDAStore=true` |
| 融合先例 | `3rdparty/cutlass/examples/13_two_tensor_op_fusion/` 与 `41_fused_multi_head_attention/` |

原始 `DefaultGemm::Mma` 用 global iterator；中间激活已在 shared memory，不能把其地址
直接传入 `ld.global`/`cp.async`。首轮新增仅有 `operator()` 的同步 threadblock MMA 编排，
用 `DefaultMma` 生成的 SmemIterator、MmaBase::SharedStorage 和原生 warp MMA 计算。
输入 iterator 模板分别选原生 global predicated 或 shared regular iterator；不自写 iterator。

原生 ScatterD 不把负索引当 mask。reflect halo 的 scatter=-1 必须在调用输出 iterator 前
判定；有效行仍由库的 predicated iterator 写回，避免 halo 覆盖/越界。
