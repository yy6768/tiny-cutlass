# GitHub candidate → 页面 → 实现抓取

恢复原版的 candidate 发现、增量筛选记录、PR diff/关键源码抓取与本地 Wiki 阅读入口。这里的 Wiki 是本 skill 的实现页，不是仓库独立的 `repo.wiki.git` 或 github.io 站点。

依赖 Python 3.10+、PyYAML 和可访问目标仓库的 `gh`。复用已有认证，不打印 token。命令从 skill 根目录执行。

## 按架构发现并抓取

```powershell
python scripts/refresh_candidate_ledger.py --architecture sm80 --architecture sm89 --repos cutlass --merged --fetch --fetch-limit 10
```

使用 `queries/search-recipes.json` 的架构词与设备别名，PR/issue 分开检索。选项：

- `--architecture` 可重复，支持 SM80/86/89/120；`--repos cutlass,vllm,flashinfer` 限定仓库。
- `--kind pr` / `--kind issue` 限定类型；`--merged` 只约束 PR；`--recipe cutlass-sm80-pr` 指定查询。
- 默认每条搜索一页（100 条）；`--max-pages 2` 扩大范围。Search 上限、截断、错误均进入 report。
- `--since 2026-01-01` 筛选更新时间；`--cutoff 2026-09-17` 在远端查询限制创建/合并日期，不重建历史 PR 状态。
- `--dry-run` 只打印计划。Search 请求默认间隔 2.1 秒；失败返回非零。

Ledger 写入 `candidates/github/<owner>--<repo>.yaml`。以 `(kind, number)` 去重，合并架构和 query 来源，保留人工 `decision/reason`。新条目为 `defer`，搜索命中不代表设备支持。`search-results.json` 记录本次搜索和抓取结果。

`--fetch-limit` 限制本次下载数，未下载数量会打印；其余 candidate 保留在 ledger。

## 抓取指定实现

```powershell
python scripts/fetch_pr_diff.py --ids pr-cutlass-1187 pr-cutlass-2177
python scripts/fetch_pr_diff.py https://github.com/NVIDIA/cutlass/issues/1181 --architecture sm86
python scripts/fetch_pr_diff.py --ledger candidates/github/NVIDIA--cutlass.yaml --decision defer include --architecture sm89 --limit 20
```

source ID 沿用现有架构路由。新 URL 可用 `--architecture` 标记目标，不自动猜架构。每次抓取写入新的时间戳目录，保留旧快照：

```text
artifacts/github/NVIDIA--cutlass/pr-2177/<UTC timestamp>/
  page.md                 上游正文、普通评论、review 和逐行评论
  issue.json / pull.json  GitHub 元数据
  comments.json / reviews.json / review-comments.json
  changed-files.json      变更文件清单（受 API 上限约束）
  diff.patch              PR 完整 diff
  key-files/head/...      固定 head SHA 的关键源码
  key-files/base/...      删除文件在 base SHA 上的内容
  PROVENANCE.yaml        来源、状态、SHA256、字节数、遗漏、错误
wiki/candidates/pr-NVIDIA--cutlass-2177.md
```

关键文件包括 CUDA/C++ headers、PTX、MLIR 和 kernel/CuTe/Triton/attention 相关 Python；测试、benchmark、CI、文档可在完整 diff 中查看。保留源码及许可证头，不执行下载内容。

NATTEN 的 Python dispatch/autotuner 也属于关键文件。大型公开 PR 遇到 GitHub diff API 的 HTTP 406 行数上限时，脚本改用 GitHub 的 `patch-diff.githubusercontent.com` 服务（不发送认证 token），校验 diff 格式并限制为 20 MiB。实际来源与预期 head/base SHA 写入 `diff_provenance`，抓取结束再核对 PR 未移动。私有 PR 的公开端点失败或其它 API 错误继续报错，不伪造完整 diff。

默认最多 40 个关键文件、单文件 1 MiB、bundle 20 MiB；`--max-files`、`--max-file-bytes` 和 `--max-bundle-bytes` 分别调整。达到限制会记录 `complete: false` 并退出非零，不以空文件伪装成功。结束时重新核对 PR head/base；移动中的 PR 报告失败，不替换 Wiki 到不一致快照。

Wiki 页列出正文、diff、SHA 和本地源码入口，并保持 `defer`。`complete: true` 仅表示所选抓取内容没有遗漏，不表示 GPU 验证通过。

## 查询和检查

```powershell
python scripts/query.py --architecture sm89 --include-candidates
python scripts/query.py --architecture sm89 --has-code
python scripts/get_page.py pr-cutlass-2177 --include-code
python scripts/get_page.py candidate-pr-NVIDIA--cutlass-2177 --include-code
python scripts/verify_captures.py
python scripts/validate.py
```

`--include-candidates` 同时检索新 Wiki 实现页并标明未审核；`--has-code` 只保留已有关键源码落盘的条目。`--include-code` 可用原 source ID 或 candidate ID 读取实际下载文件。`verify_captures.py` 检查来源清单中的文件、长度和 SHA256；`validate.py` 检查维护页面、链接和人工资料索引。抓取不覆盖人工摘要。
