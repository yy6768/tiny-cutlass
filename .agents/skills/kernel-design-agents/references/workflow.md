# KDA 在 tiny-cutlass 中的实验工作流

适配自固定版本的 [上游 agent flow](upstream/docs/agent-flow.md)。本仓库的代码、构建和
reference 规则优先；上游“另建 workspace”的原则在这里体现为 family/实验范围清楚。
是否使用 worktree 取决于任务，不能为隔离而丢弃或重置已有修改。

## 从契约到候选

1. **契约**：写出操作语义、shape/layout/dtype、目标设备、reference、容差、性能/质量
   指标和晋升条件。研究与硬件信息缺失可先标记；不要把空字段填成默认“已支持”。
2. **基线**：读根和 family 的 AGENTS，检查现有代码/脚本，确认 reference 和验证入口。
   记录 git commit 与相关未提交 patch；仅有 commit 不能复现工作树修改。
3. **调研与 draft**：读取对应架构页和必要源码，把候选、依据、风险、预期改变的
   指标、验证方式写入 family 的设计文档。
4. **可执行 plan**：明确改哪些文件、实例化哪些 policy、如何验证和计时、何时停止某候选。
   由同一 agent 将 draft 收敛成 plan 即可；不依赖 Humanize 插件。
5. **实现**：每轮围绕一个可区分的假设。多个因素必须一起变更时在记录中解释。
6. **验证**：先 build、再 reference parity；失败只记录失败原因，修正后重跑必要验证。
7. **测量与解释**：普通 benchmark 比较候选；必要时用 NCU/Systems 解释机制。
8. **取舍**：依据契约采用、继续修改或淘汰，保留证据和回归限制。

## 路径映射

以下 `<family>`、`<experiment>`、`<run-id>` 是模板字段，实际命令必须替换。

| 内容 | 当前仓库约定 |
|---|---|
| 设计与计划 | `csrc/<family>/docs/<experiment>/draft.md`、`plan.md`；已有文档布局则沿用 |
| kernel / launcher / binding | family 内按现有职责分层；core 不持有 Torch tensor |
| harness / reference / benchmark | `csrc/tests/<family>/`；`verify.py`、`bench.py`、短 C++ 文件和 target 名 |
| 脚本入口 | `scripts/kernels/<family>/<family>.bat`，严格 build → verify → bench |
| 所有构建产物 | `build/<family>/<experiment>/...` |
| 一次实验的机器输出 | `build/experiments/<family>/<run-id>/` |
| `.ncu-rep` / CSV / 分析 | 上述 run 目录中的 `profile/`、`analysis/` |
| 可维护的结论 | family 文档链接原始证据，注明缺失/未提交的本地产物 |

已有 family 把 profiler 报告放在忽略的 `profile/<run-id>/` 时可继续沿用，但可执行文件、
obj 等构建产物仍只能在 `build/`。不要采纳上游 NCU 技能“编译到 profile/harness”的路径。
新 run 使用新目录，baseline 和 candidate 使用不同输出名，不覆盖早期证据。

## 最小实验记录

推荐 `environment.json`、`candidates.jsonl`、`benchmark.csv` 和原始日志，或沿用
family 已有等价格式。字段以可复现为准：

- candidate id、parent id、源码 revision/patch、改变的 policy、假设。
- GPU UUID/名称/CC、软件版本、编译目标、命令、输入标识/shape、dtype/layout。
- 验证 reference、容差、误差、通过/失败/未运行和日志路径。
- 计时范围、warmup/repeats、graph 模式、median/波动、baseline、workspace/峰值显存。
- report 路径、实际 kernel、关键 metric 的值和单位、采用或淘汰理由。

缺失数值写 `null` / `not-run`，不能写 0 或 pass。失败候选可有编译或错误日志，
不能附上未经 parity 的数据并将其标成有效性能。

## 晋升标准

候选须通过完整契约内验证，并有目标设备上同条件的性能/资源证据；Diffusion 替换再通过
[模型验收](diffusion.md)。噪声内变化不称作加速；超出契约的 shape 不能悄悄接入 fallback。
如果只有局部 shape 提升，明确支持范围与 dispatch 条件，不用整体平均值掩盖重要回归。

## 多 agent 接续

本技能不自动开启并行 agent。获得并行任务要求时，可把源码调研、候选实现、独立 reference
检查交给不同 agent。每份交接包含目标、架构/工具版本、文件范围、命令和证据路径。
实现 agent 不通过修改 reference 容差来接受自己结果；集成人核对源码与证据一致。
共用 GPU 时串行 benchmark/NCU，避免竞争污染数据。人写后记、博客 Overview 的规则仍按
仓库现有 `cutlass-blog-workflow` 执行。
