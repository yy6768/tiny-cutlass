---
name: rtx-kernel-wiki
description: 为 NVIDIA SM80/SM86（Ampere）、SM89（Ada）和 SM120（RTX Blackwell）的 CUDA/CUTLASS/CuTe/Triton kernel 实现与调优检索架构资料、源码、PR、issue 和实现博客。用于 GEMM、attention、卷积融合、量化及访存问题；不覆盖以 SM90/SM100 为目标的实现或框架部署。
---

# RTX Kernel Wiki

面向 **SM80、SM86、SM89、SM120** 的 kernel 实现资料库。先按实际 GPU、算子和数值要求找源码，再用 PR/issue 定位实现差异与已知问题。

这是人工筛选的资料快照，不是上游全量镜像。每页 `retrieved_at` 标明核查日期；`architectures` 表示检索相关性，**不等于该 kernel 已在这些设备验证通过**。本库没有本地 GPU parity 或性能认证。

## 实现前先确定架构

阅读 [架构与实现入口](references/architectures.md)，确认 GPU compute capability、CUDA/CUTLASS 版本、dtype、layout、shape、对齐与 shared-memory 预算。

- **SM80**：A100/A30；从 `mma.sync`、`ldmatrix`、`cp.async`、多级流水和 CUTLASS 2.x/CuTe 实现出发。
- **SM86**：RTX 30 系列、A10、RTX A6000；独立检查 tile/stages 的资源占用。A100 上能启动的配置不代表能在 SM86 启动。
- **SM89**：RTX 40 系列、L4/L40/L40S；延续 warp MMA 路线，FP8 使用对应 Ada atom 和缩放语义。区分 CUTLASS policy 的 ArchTag 与 nvcc 实际编译目标。
- **SM120**：RTX 50 系列、RTX PRO Blackwell；使用 warp-level MMA 和该架构的低精度路径。TMA 可用于 SM120，但不能把 SM100 的 `tcgen05`/TMEM kernel 直接迁入。`sm_120`、`sm_120a`、`sm_120f` 的条件逐项核对工具链和上游源码。

不要把 SM121（DGX Spark）、SM100（B200）或 SM90（H100）映射成 SM120。跨架构资料只有明确的可迁移部分才可引用；不能把历史 issue 的标题当成当前硬件限制。

本 skill 提供实现依据。真正修改 CUDA/CUTLASS kernel 时先使用仓库的 `cutlass-kernel` skill，并遵守所在目录 `AGENTS.md`：policy 模板化，核心入口使用 device pointer/problem/stream，build → verify → bench；不支持的 TensorOp 配置明确失败。

## 检索入口

以下命令从本 skill 目录运行；也可在任意目录用脚本绝对路径调用。依赖 Python 3.10+ 和 `requirements.txt` 中的 PyYAML。

需要更新资料时使用 [GitHub 抓取流程](references/github-capture.md)。`refresh_candidate_ledger.py` 恢复原版 candidate 发现与增量 ledger；`fetch_pr_diff.py` 拉取 PR/issue 正文、讨论、完整 diff 和固定 SHA 的关键源码，并生成本地 Wiki 实现页。联网抓取使用已认证的 GitHub CLI `gh`。

```powershell
python scripts/refresh_candidate_ledger.py --architecture sm80 --architecture sm89 --repos cutlass --merged --fetch --fetch-limit 10
python scripts/fetch_pr_diff.py --ids pr-cutlass-1187 pr-cutlass-2177
python scripts/query.py --architecture sm89 --include-candidates
python scripts/get_page.py candidate-pr-NVIDIA--cutlass-2177 --include-code
```

```powershell
python scripts/query.py --architecture sm80 --type pr --tag cp-async
python scripts/query.py --architecture sm86 "shared-memory"
python scripts/query.py --architecture sm89 --type pr --status merged --tag fp8
python scripts/query.py --architecture sm120 --type issue --tag nvfp4
python scripts/query.py --architecture "RTX 4090" --type blog
python scripts/get_page.py pr-cutlass-2177
python scripts/grep_wiki.py "TMA|scale" --architecture sm120
```

`--architecture` 接受 SM 编号和已列明的 GPU 别名；裸 `Blackwell` 有歧义，不自动映射。`--type` 区分 `pr/issue/blog/doc`。`--status` 区分 `merged/open/closed-unmerged/closed/published`。其他筛选见 `--help`；文本查询是关键词匹配，不是语义搜索。

按问题直接阅读索引：

- [SM89 Window / Neighborhood Attention 候选](references/sm89-local-attention.md)：4070/4090、CUTLASS 2.x 与 FP8 的首批 merged PR、源码抓取及待补量化契约。
- [SM80](queries/sm80.md)、[SM86](queries/sm86.md)、[SM89](queries/sm89.md)、[SM120](queries/sm120.md)：按架构找 PR、issue、博客和官方文档。
- [按问题](queries/by-problem.md)、[按技术](queries/by-technique.md)、[按算子](queries/by-kernel-type.md)、[按仓库](queries/by-repo.md)。
- [上游 PR/issue 查询](queries/github-search.md)：带架构限定的可复用 GitHub 搜索式。
- [实现取证流程](references/implementation.md)：从候选源码到本地实验的检查点。

## 如何使用证据

1. 读取候选页的适用理由、限制、版本、源码链接与后续讨论。PR 的 `head_sha` 固定候选实现；合并状态通过 `merged_at` 核查，closed 不等于 merged。
2. 优先检查已合并 PR 和官方示例。open/closed-unmerged PR 仅作实现候选；issue 是复现线索，关闭也不能单独证明已修复。
3. `source-reported` 表示来源陈述；`inferred` 表示可迁移性推断。两者都不是本地验证。博客中的其它 GPU 数据不得转成目标卡性能结论。
4. 回答实现问题时给出 **目标架构 → 实现选择 → 源码位置/固定 SHA → 适用条件 → 验证办法**，引用具体本地页及上游 URL。必要时实时重查上游，尤其是未合并 PR、编译器回归和版本支持。
5. 性能数字只有在 GPU、dtype、shape、metric/value、软件版本及来源齐全时才引用；明确区分上游测量与本地复现。无法运行目标设备时直接标为未验证。

## 维护

阅读 [筛选与更新规则](references/maintenance.md)。`sources/` 是人工审核后的有效资料；`candidates/` 记录检索结果和筛选决定；`artifacts/github/` 保存正文、diff、关键源码与清单；`wiki/candidates/` 是自动生成的实现阅读页。抓取结果保留 candidate 状态，通过 `--include-candidates` 查询。

```powershell
python scripts/refresh_candidates.py --list
python scripts/refresh_candidates.py --recipe cutlass-sm120-pr --max-pages 1 --output ../../../build/rtx-kernel-wiki-candidates.json
python scripts/generate_queries.py
python scripts/validate.py
python -B -m unittest discover -s tests -v
```
