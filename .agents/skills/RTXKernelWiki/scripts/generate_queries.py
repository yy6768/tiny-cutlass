"""Rebuild deterministic Markdown indices from curated source metadata."""
import json
from collections import defaultdict
from urllib.parse import quote
from catalog import ROOT, ARCHITECTURES, load_pages


def entry(page):
    m = page['meta']
    title = m['title'].replace('|', '/').replace('\n', ' ')
    return f"- [{m['id']} — {title}](../{page['path']}) · {m['status']} · {', '.join(m['architectures'])}\n  {m['inclusion_reason']}"


def render_groups(title, pages, key):
    groups = defaultdict(list)
    for p in pages:
        values = p['meta'].get(key, [])
        if isinstance(values, str):
            values = [values]
        for value in values:
            groups[value].append(p)
    text = f'# {title}\n\n由 `scripts/generate_queries.py` 生成。架构标签表示相关性；状态与本地验证分开记录。\n'
    for name, group in sorted(groups.items()):
        text += f'\n## {name}\n\n' + '\n\n'.join(entry(p) for p in group) + '\n'
    return text


def rendered_indices(root=ROOT):
    pages = load_pages(root)
    out = {}
    for arch in ARCHITECTURES:
        subset = [p for p in pages if arch in p['meta']['architectures']]
        out[f'{arch}.md'] = render_groups(f'{arch.upper()} 实现资料', subset, 'type')
    for name, title, key in [
        ('by-architecture', '按架构', 'architectures'),
        ('by-problem', '按问题', 'symptoms'),
        ('by-technique', '按技术', 'tags'),
        ('by-kernel-type', '按算子', 'kernel_types'),
        ('by-repo', '按仓库', 'repo'),
        ('by-status', '按上游状态', 'status'),
    ]:
        out[f'{name}.md'] = render_groups(title, pages, key)
    recipes = json.loads((root / 'queries/search-recipes.json').read_text(encoding='utf-8'))
    text = '# GitHub PR / issue 搜索\n\n这些是发现入口，不是收录或支持证明。别名分开搜索；结果按正文、代码和讨论审核。\n\n'
    for r in recipes:
        url = 'https://github.com/search?type=issues&q=' + quote(r['query'], safe='')
        text += f"- **{r['id']}** ({r['architecture']}, {r['kind']}) — [上游搜索]({url})\n  `{r['query']}`\n\n"
    out['github-search.md'] = text
    return out


def main():
    out = rendered_indices()
    for name, text in out.items():
        (ROOT / 'queries' / name).write_text(text, encoding='utf-8', newline='\n')
    print(f'Generated {len(out)} indices from {len(load_pages())} curated sources.')


if __name__ == '__main__':
    main()
