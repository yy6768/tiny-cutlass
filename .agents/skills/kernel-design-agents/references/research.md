# 架构与算子证据检索

KDA 的调研只为当前候选回答具体问题。先读本仓库的 family 实现，再读固定版本的
`3rdparty/cutlass`，然后补官方 CUDA/PTX/NCU 文档或原始论文。最终验证由目标设备完成。

## KernelWiki 的用法与边界

上游 KDA 在 gitlink 中固定
[KernelWiki `76d27b5`](https://github.com/DongyunZou/KernelWiki/tree/76d27b56f804e7e7295d4c570e1e5d7eef4b0a75)。
该快照的 SKILL 声明主要面向 SM100/B200 和 SM90，资料截至 2026-04-27。
本适配不复制或改写其数千个 PR 快照；需要时沿固定链接追溯原文，不能把架构标签批量改成 SM120。

如已有完整 KernelWiki 安装，可在其目录按需运行其只读检索：

```text
python scripts/query.py --architecture sm90 --limit 10
python scripts/query.py "epilogue fusion" --limit 10
python scripts/get_page.py <实际检索得到的 page-id> --follow-sources
```

这些命令属于外部 KernelWiki，不是本技能安装的脚本。没有外部安装也可以完成 KDA：
直接查当前 CUTLASS 与官方资料即可。SM89/SM120 的检索为空时，不能退而接受无架构匹配
的 SM100 代码；用对应 Ada/GeForce 示例补证。

## 各架构的检索线索

| 目标 | 有用关键词 | 读取结果后核对 |
|---|---|---|
| SM89 | Ada、`mma.sync`、`cp.async`、example 58、epilogue | 文件 arch 条件、layout、实际设备 |
| SM90 | Hopper、TMA、WGMMA、warp specialization、example 48/49 | 90a、barrier 生命周期、shared-memory 需求 |
| SM120 | Blackwell GeForce、`mma_sm120`、block_scale、example 79 | 120a/120f、scale 格式、单 CTA cluster |
| Diffusion | DiT/U-Net、AdaLN、non-causal attention、norm modulation | 真实模型语义、shape、timestep、端到端成本 |

## 引用与证据状态

- 官方 capability 说明只能证明能力；源码能证明某版本的实现和限制。
- PR / Wiki 中的性能按 source-reported 记录：GPU、dtype、shape、metric、数值、原始来源。
  这不是本仓库的性能结果。
- 本机编译、reference parity、benchmark、profile 各有独立证据路径和日期。
- 工程推断写明依据和待验证条件；不得把推断升级成 measured。
- 引用上游代码前核对许可证、commit、构建条件和失败/skip 行为。更新依赖不能覆盖本地修改。

原始上游快照只用于理解来源；本技能的执行入口是 [SKILL.md](../SKILL.md)。
