---
name: kernel-design-agents
description: "在 tiny-cutlass 中使用 KDA 工作流研究、规划、验证和优化 SM89、SM90、SM120 CUDA/CUTLASS 算子，重点覆盖端侧 Diffusion 的 attention、GEMM/MLP、卷积、归一化与融合。用于架构选型、优化计划、候选实验和性能证据整理；实际 kernel 开发同时使用 cutlass-kernel。"
---

# Kernel Design Agents

这是 [mit-han-lab/kernel-design-agents](https://github.com/mit-han-lab/kernel-design-agents)
在当前仓库的中文适配版。继承“任务契约 → 调研 → 计划 → 候选 → 验证 → 测量 → 取舍”的循环。
安装内容、固定版本和上游依赖的处理见 [来源与维护](references/provenance.md)。

## 先读什么

首次使用读 [架构总览](references/architecture-overview.md)，然后只加载本次目标架构：

| 任务 | 按需阅读 |
|---|---|
| 当前 Ada / SM89 | [SM89](references/sm89.md) |
| Hopper / SM90，含 `sm_90a` 特性目标 | [SM90](references/sm90.md) |
| Blackwell RTX / SM120，含 `sm_120a` / `sm_120f` | [SM120](references/sm120.md) |
| 端侧 Diffusion 热点和验收 | [Diffusion 优化路线](references/diffusion.md) |
| 开始/继续候选实验、接续其他 agent 的工作 | [实验工作流](references/workflow.md) |
| NCU 采集、报告分析、瓶颈诊断 | [三架构 Profiling](references/profiling.md) |
| 查论文、KernelWiki、CUTLASS 源码 | [证据检索](references/research.md) |
| 用户准备提出具体优化任务 | [任务模板](references/task-template.md) |

## 仓库约束

1. 读仓库根和目标 family 的 `AGENTS.md`。实际 kernel 开发先使用仓库现有
   [cutlass-kernel](../../../.codex/skills/cutlass-kernel/SKILL.md)。学习博客另走
   `cutlass-blog-workflow`；这些参考文档不替代博客的人工确认流程。
2. 当前配置开放 SM80/SM89，默认 SM89；SM90/SM120 是本技能覆盖的迁移目标，
   不代表本仓库 kernel 已完成支持。重新检查 CMake、工具链、具体 policy 和实际 GPU。
3. primary operator / policy / target / 公共 API 保持模板化，架构、dtype、layout、tile
   放在模板参数、显式实例化、launcher 或 CMake 配置层。不要添加 concrete 旧名 alias。
4. TensorOp 配置不支持时明确失败，不能静默退到 SIMT/raw CUDA。独立的 norm、索引、
   pointwise 算子可按语义选择实现；不要把它们误称为 TensorOp fallback。
5. 核心入口采用 raw device pointer、problem descriptor、`cudaStream_t`；Torch ownership
   留在 binding / reference 层。
6. build → verify → bench；verify 失败则停止 bench / NCU。构建输出全部在 `build/`，
   harness 在 `csrc/tests/<family>/`，使用短测试/target 名、`verify.py` 和 `bench.py`。

## 执行方式

- 先明确模型/算子、真实 shape、GPU、精度、reference、容差和目标指标。缺失信息写成
  待确认项，先推进代码和资料检查；不要编造模型、测量或验收结果。
- 为实际优化任务写 family 内的 draft 和可执行 plan，再开始候选实现。普通文档问答无需
  创建实验目录；已有计划可直接接续。技术计划不额外要求用户逐步批准。
- 每次候选回答一个可检验的问题；验证后在相同输入、计时方式和设备状态下与 baseline 比较。
  记录失败候选、父候选、命令、源码版本和证据路径。
- 将“官方能力”“源码存在”“本机编译通过”“reference 通过”“本机实测收益”分别记录。
  SM90、SM120 没有目标设备验证时，继续可做的设计/编译检查，交付中标明运行待验证。
- 不将 SM100/B200 的 `tcgen05`、TMEM、2-SM、HBM 参数或 NCU metric 名套到 SM120。
  不将 SM90 的 WGMMA 二进制视为 SM120 可运行实现。
- KDA 可以由单个 agent 执行。用户或适用规则明确要求并行时，再按调研、实现、验证划分
  有界子任务；共享 GPU 的计时/NCU 串行进行，证据随任务交接。
- 结束时给出改动、reference 状态、实测范围、候选取舍和未解决限制。

可直接调用：`使用 $kernel-design-agents，为 SM89 上的 Diffusion attention 建立优化计划。`
