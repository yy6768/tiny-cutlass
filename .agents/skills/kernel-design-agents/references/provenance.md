# 安装来源与维护

安装日期：2026-09-09。安装位置：仓库根 `.agents/skills/kernel-design-agents/`。
调用名：`$kernel-design-agents`；保留默认自动匹配，并提供 `agents/openai.yaml` 元数据。
仓库级发现目录依据 [OpenAI 官方技能文档](https://learn.chatgpt.com/docs/build-skills#where-codex-loads-local-skills)。

## 固定来源

| 来源 | 固定版本 | 本次处理 |
|---|---|---|
| [mit-han-lab/kernel-design-agents](https://github.com/mit-han-lab/kernel-design-agents) | `9f5ee5d5cd623c2d69c4ff71c8a5155edcb8c43b` | 将工作流包装成当前仓库可发现的 skill，并新增三架构与 Diffusion 参考文档 |
| [KernelWiki gitlink](https://github.com/DongyunZou/KernelWiki/tree/76d27b56f804e7e7295d4c570e1e5d7eef4b0a75) | `76d27b56f804e7e7295d4c570e1e5d7eef4b0a75` | 作为可选外部资料入口；不安装完整 Wiki、不改其历史架构标签 |
| [ncu-report-skill gitlink](https://github.com/DongyunZou/ncu-report-skill/tree/d1887948c7d53690cfe6605f59c1329b8a1c6bb5) | `d1887948c7d53690cfe6605f59c1329b8a1c6bb5` | 参考其证据流程，在本技能内重写三架构 profiling 指南；不另装 B200 默认技能 |

KDA 上游根目录没有 `SKILL.md`，不能把仓库 URL 当作现成 skill 路径直接交给
`install-skill-from-github.py`。本次依 `skill-installer` 检查来源和结构，使用
`skill-creator` 的初始化器创建总入口，再移植并适配工作流。
上游 README 中的 Humanize 是其 Claude 使用建议，本地计划流程不依赖它。

本技能可独立执行研究/计划流程，实际 CUDA 工作使用本仓库现有工具链和 `cutlass-kernel`。
没有修改用户全局 skill、插件或个人配置，也没有给根 CMake 添加新架构支持。

## 保留的上游文件

- [原始 README](upstream/README.md)。
- [原始 agent-flow](upstream/docs/agent-flow.md)。
- [原始 basic-flow prompt](upstream/prompts/basic-flow.md) 与 [prompt 用法](upstream/prompts/README.md)。
- [原始 CLAUDE.md](upstream/CLAUDE.md)，仅保留来源上下文，规则不用于当前仓库。
- [原始 MIT LICENSE](../LICENSE)，保留 mit-han-lab 的版权声明。

这些文件按固定 commit 原样保留，用于追溯来源，不是当前执行入口。
其中的路径相对上游原仓库，查阅未随安装复制的文件时使用
[固定 commit 树](https://github.com/mit-han-lab/kernel-design-agents/tree/9f5ee5d5cd623c2d69c4ff71c8a5155edcb8c43b)。
本地适配保留 KDA 契约、draft/plan、候选关系和证据晋升规则，改用中文、仓库路径与
Windows 工作流，并增加 SM89/SM90/SM120 能力边界、Diffusion 质量与性能验收。

## 后续更新

1. 在临时/忽略目录检出上游新 commit，与固定来源比较；先检查子模块 gitlink 是否变化。
2. 对照当前 CUTLASS/CUDA/NCU 更新架构页，逐项核对 dtype、指令和 compiler target。
3. 保留本地契约和 provenance；不要以更新为由覆盖本仓库未提交内容。
4. 更新来源 SHA 和日期，校验 SKILL frontmatter、引用路径及示例参数。
5. 新增 kernel 支持另做 build/parity/bench；文档更新不能自动改变支持状态。

安装时只做技能结构与文档检查，未产生三架构 GPU benchmark 或 NCU 性能结论。
