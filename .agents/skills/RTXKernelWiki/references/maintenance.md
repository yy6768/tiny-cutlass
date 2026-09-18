# 筛选与更新规则

本次重置把旧 SM100/Hopper 库整体退出有效目录。来源与检索日期按条目记录，不宣称全量覆盖或统一的“知识截止”。`data/refresh.yaml` 记录本次快照范围；`candidates/selection.md` 记录误导性或不适用来源的淘汰理由。

## 收录门槛

- 目标为 SM80/86/89/120 中至少一种，有具体 kernel、指令、布局、pipeline、融合或复现问题。纯模型部署、路由、分布式通信和泛新闻不收录。
- PR 必须读取正文、相关讨论和 changed files；通过 Pull API 的 `merged_at` 判断是否合并。记录 head SHA、变更文件数量、可读源码链接和 `open/merged/closed-unmerged`。被关闭或替代的补丁只能标作历史候选，不能称为上游已实现。
- issue 单独放在 `sources/issues/`，记录 open/closed、适用版本与复现线索；不能把 closed 或标题“fixed”当作合并证明。撤回的根因解释进入淘汰记录。
- 博客必须有可用的实现内容/代码入口或足以指导实现的官方解释；写清直接适用、可迁移部分、未验证部分和错误/过时细节。不复制原文或整段代码，不把 B200/H100 benchmark 改标签成 RTX。
- `confidence` 只使用 `source-reported/inferred`；本快照所有 `local_validation` 为 `not-run`。来源声称测试成功时，仍标明测试者和设备，不能升级为本地 verified。

## 数据格式

`sources/{prs,issues,blogs,docs}/*.md` 是默认的人工检索语料。每页 YAML frontmatter 包含 `id/title/type/url/architectures/architecture_evidence/tags/kernel_types/symptoms/status/confidence/retrieved_at/inclusion_reason/local_validation`。PR/issue 另有 `repo/number/updated_at`；PR 另有 `head_sha/merged_at/changed_files/code_urls`。`architectures` 是路由标签，`architecture_evidence` 解释逐项适用理由和推断范围。

新增来源时先核对 URL、状态、实现与讨论，再写简短中文摘要。保留具体限制和源码入口。若 PR 已无 head repository，选择仍可访问的固定 commit/diff 并说明；不要编造 raw 源文件 URL。

## 搜索与重建

完整的原版式 candidate 抓取入口是 `refresh_candidate_ledger.py --fetch`：更新 ledger、抓取上游页面/diff/关键源码、生成 `wiki/candidates/`。见 [GitHub 抓取流程](github-capture.md)。下方 `refresh_candidates.py` 保留为仅输出搜索 JSON 的轻量入口。

`queries/search-recipes.json` 是 GitHub 搜索式；`refresh_candidates.py` 使用已登录的 `gh api` 只读查询，结果写入指定的候选 JSON。脚本记录每条 query 的 total_count、实际页数、是否截断、是否 incomplete、错误与抓取时间。默认只抓一页；GitHub Search 最多返回 1000 条，不把部分结果描述成全量。

```powershell
python scripts/refresh_candidates.py --list
python scripts/refresh_candidates.py --recipe cutlass-sm80-pr --recipe cutlass-sm89-issue --max-pages 2 --output ../../../build/rtx-candidates.json
```

搜索别名应覆盖 `sm80/sm_80/A100`、`sm86/RTX 3090/A10`、`sm89/RTX 4090/L40S`、`sm120/sm_120/RTX 5090/RTX PRO Blackwell`；搜索发现不等于架构归属。需要更大范围时新增独立 recipe，不用裸 `Blackwell` 扫库。增量搜索可使用 `updated:>=YYYY-MM-DD`，但不要丢失基线时期的旧修复。

审核候选并修改 source 页后运行：

```powershell
python scripts/generate_queries.py
python scripts/validate.py
python -B -m unittest discover -s tests -v
```

生成脚本仅重建它管理的 Markdown 索引，不改人工 source 页。校验检查元数据、链接、固定 SHA、状态一致性及索引是否过期；它不验证 GPU 正确性。`--include-candidates` 可额外搜索抓取后生成的 Wiki 页。原始 artifacts、旧缓存、备份和 `build/` 不作为默认搜索语料。
