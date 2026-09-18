# SM89 Window / Neighborhood Attention 候选

筛选日期：2026-09-18。目标是 RTX 4070/4090、CUTLASS 2.x 风格实现，并为 FP8 接入找依据。架构映射见 [NVIDIA GPU 表](https://developer.nvidia.com/cuda/gpus)。本轮只做资料筛选和源码抓取，GPU 验证为 `not-run`。

13 条 merged 查询覆盖 7 个仓库，得到 **244 个去重命中**，没有搜索截断；进一步核查 23 个 PR 的正文和 changed files，首批收录 **10 个 merged PR**，其余候选保留待审或有理由排除。查询命中包括 Windows、SM100 等噪声，不能把全部命中算成 SM89 实现。

[本轮筛选清单](../candidates/sm89-local-attention.json) 保存角色、优先级、merged_at、固定 head、源码目录和排除理由；[搜索原始结果](../candidates/sm89-local-attention-search.json) 保留全部查询。`include` 只表示值得研究，不表示批准迁入 kernel。

## 先看这几条

| 顺序 | merged PR 与本地人工页 | 提供什么 | 采用边界 |
|---|---|---|---|
| 1 | [NATTEN #111](../sources/prs/natten-111.md) | FNA 的 1D/2D/3D 邻域 mask、KV tile 范围、融合 forward | FP16/BF16/FP32 路线；邻域窗口与本地 reflect 分块不同 |
| 2 | [CUTLASS #828](../sources/prs/cutlass-828.md)、[#992](../sources/prs/cutlass-992.md) | example 41 的 QK→online softmax→PV、bias 与 shared-memory MMA | 2.x attention 主线参考；没有完成目标 FP8 全链 |
| 3 | [vLLM #6677](../sources/prs/vllm-6677.md) | Ada FP8 GEMM 的 tile、stage、M/N 分派和 scale epilogue | 只是 FP8 primitive，不能替代完整 attention |
| 4 | [NATTEN #114](../sources/prs/natten-114.md) | SM86/89 排除过大的 tile 配置 | 作者当时未实测 SM89；需重新核算具体实例 |
| 5 | [NATTEN #337](../sources/prs/natten-337.md) | 64 位 token strides、较新 FNA forward 完整文件 | 寻址修正，不是 FP8 支持 |
| 6 | [SageAttention #196](../sources/prs/sageattention-196.md) | INT8 QK、FP8 PV、混合累加 | CUDA/inline PTX 机制参考，不进入当前 2.x 主线 |
| 7 | [CUTLASS #3278](../sources/prs/cutlass-3278.md) | ScatterD iterator 指针推进修正 | 仅使用该 iterator 的路径相关 |
| 8 | [FasterTransformer #333](../sources/prs/fastertransformer-333.md) | WindowAttention 的 QK scale 接口修正 | FP16/TRT 路线，只参考数值语义 |
| 9 | [NATTEN #341](../sources/prs/natten-341.md) | backward partial-tile mask 与 dV NaN 修复 | 当前本地 forward 不受这条 backward 修复直接影响 |

每个人工页均链接固定 SHA 源码、下载文件及自动 Wiki。自动 Wiki 保存原始 PR 正文、普通评论、review、逐行评论、完整 diff 与 SHA256 清单；它的默认未审核标记不覆盖人工页的筛选结论。

## 实现前要保留的差异

**Window 与 Neighborhood 分开建契约。** 本地 Swin 是 window4、L=16、reflect/RMSNorm、预展开 bias、QKV/proj 融合；本地 NATTEN 是非 causal 的 1D forward、连续 BLHD、stride=dilation=1。FNA 的滑动邻域不能直接当成 Swin 分块窗口，1D sliding-window attention 也不能直接替换 2D 空间邻域。参照 [Swin 约束](../../../../csrc/swin/AGENTS.md) 和 [NATTEN 约束](../../../../csrc/natten/AGENTS.md)。

**bias/scale 顺序已经发现差异。** CUTLASS #828 的固定 head 在 `kernel_forward.h:765-796` 先乘 scale 再加 bias，即 `QK*scale+bias`；本地要求 `(QK+bias)*scale`。复用组件时必须保留本地 reference，不能通过改 reference 或放宽容差掩盖差异。

**2.x 指 API/组件路线。** 本地 `3rdparty/cutlass/include/cutlass/version.h` 当前为 4.5.2，但保留 threadblock/warp/iterator/epilogue 等 2.x 组件。vLLM #6677 使用 2.x GEMM/EVT，同时包含 CuTe 类型工具；这不等于 CuTe-DSL kernel。已有 [CUTLASS 实现入口](../sources/docs/cutlass-implementation-map.md) 的 example 58 和 `include/cutlass/arch/mma_sm89.h` 可补充 Ada FP8 primitive；不要用 CuTe atom PR 代替 2.x kernel 证据。

**量化语义不能从 GEMM 自动推导。** vLLM #6677 的 scale 是 A 的 per-tensor/per-row 与 B 的 per-tensor/per-column，zero point 为 0。它不确定 attention 的 QK/PV/proj 是否量化，也不提供沿 GEMM K 维的 block scaling 契约。SageAttention #196 的 QK 实际用 INT8；FP8 用于 PV，不能标成全 FP8 attention。

本轮已核查候选中，尚未确认一条同时满足 **SM89 + 当前 Window/Neighborhood 语义 + FP8 + CUTLASS 2.x 完整融合** 的现成实现。这是本轮筛选的证据缺口，不是断言上游不存在这样的实现。

## 排除与延后

- [NATTEN #266](https://github.com/SHI-Labs/NATTEN/pull/266) 虽然标题是 FP8 FNA，但 [固定版本检查](https://github.com/SHI-Labs/NATTEN/blob/26242c2d7cd9e9c81a19370edfa41cc2e8eb1668/src/natten/backends/configs/checks.py#L70) 限定 SM100/SM103，排除 SM89。
- [SageAttention #226](https://github.com/thu-ml/SageAttention/pull/226) 的 attn_mask 进入 Triton INT8-QK/FP16-PV 路径，不能据此声称 CUDA FP8-PV 支持目标邻域 mask。
- [FlashAttention #2624](https://github.com/Dao-AILab/flash-attention/pull/2624)、[#1856](https://github.com/Dao-AILab/flash-attention/pull/1856) 修改 CuTe Python 路线，不进入当前 2.x 主线；前者可参考 0 与 None 的窗口边界区别。
- [xFormers #362](https://github.com/facebookresearch/xformers/pull/362)、[#587](https://github.com/facebookresearch/xformers/pull/587) 是有价值的历史来源；优先读取已经汇入 CUTLASS example 41 的 #828/#992，暂不重复抓取绑定与生成实例。
- [FasterTransformer #242](https://github.com/NVIDIA/FasterTransformer/pull/242) 是 SM86 INT8；[vLLM #6384](https://github.com/vllm-project/vllm/pull/6384) 是早期禁用未调优 Ada GEMM 的策略，不作为本轮实现方案。

其余命中保持 `defer`，没有逐一完成代码审核；不可用未审核数量声称覆盖了全部实现。

## 用户下一步主导什么

建议先推进现有 **window4 attention 的 FP8 契约**，把 NATTEN 作为独立的邻域实现参考。现有边界语义已记录，无需重新发明默认约定；需要用户决定的是以下输入：

| 用户需要确定 | 要交给实现的具体内容 |
|---|---|
| 代表性 workload | GPU 的具体型号、B/H/W/C/G/Dq/Dv、shared/separate QK、shift、各 shape 出现频率 |
| FP8 范围 | QKV、QK、P/V、proj 各阶段 operand/output dtype，E4M3/E5M2，哪些阶段保留 FP16 |
| 量化 reference | 可运行参考与真实输入/权重；scale 编码/解码方向、粒度、动态/静态与生命周期、舍入/饱和及异常值规则 |
| 验收 | 对量化 reference 的误差、相对原模型的质量要求，以及是否计入量化/scale 更新/准备成本的性能 baseline |

这些项目已有 [FP8 实施前必填契约](../../../../csrc/swin/docs/02-window-attention/plan.md)。资料筛选不依赖它们全部填写；接入 FP8 kernel 前需要齐备。后续顺序是：确定契约 → 选择可复用组件 → 小范围候选实现 → 完整 reference parity → 4070/4090 各自 bench/NCU。完整 attention 和完整 block 分开验收。

## 复现检索与下载

以下命令从 skill 根目录执行。检索先写 ledger，再按人工决定下载，避免按编号抓到无关的旧候选。

```powershell
python scripts/refresh_candidate_ledger.py --architecture sm89 --repos NATTEN --kind pr --merged
python scripts/query.py --architecture "RTX 4070" --type pr --status merged --tag cutlass-2x
python scripts/query.py --architecture sm89 --type pr --status merged --tag fna
python scripts/get_page.py pr-natten-337 --include-code
python scripts/get_page.py pr-vllm-6677 --include-code
python scripts/fetch_pr_diff.py --ledger candidates/github/SHI-Labs--NATTEN.yaml --decision include --architecture sm89 --max-files 300 --max-file-bytes 8388608 --max-bundle-bytes 67108864
python scripts/verify_captures.py
```

本轮 FNA 起始 PR #111 超过 REST diff 行数上限，最终使用 GitHub 的公开 patch 服务。原失败尝试单独归档，成功抓取仍必须校验固定 head/base、文件长度与 SHA256。`--max-files 300` 用于保留其 202 个关键源码及生成实例；其中一个 autogen header 超过默认 1 MiB，因此此例显式提高单文件和 bundle 上限。默认限制不足时会明确标为不完整。
