"""Shared, deliberately scoped source loader for RTX Kernel Wiki."""
from pathlib import Path
import re
import sys
import yaml

# Keep redirected CLI output readable on Windows regardless of the ANSI code page.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = ('sm80', 'sm86', 'sm89', 'sm120')
TYPES = ('pr', 'issue', 'blog', 'doc')
STATUSES = ('merged', 'open', 'closed-unmerged', 'closed', 'published')


def read_page(path):
    text = path.read_text(encoding='utf-8-sig')
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)', text, re.S)
    if not match:
        raise ValueError(f'Missing frontmatter: {path}')
    meta = yaml.safe_load(match[1])
    if not isinstance(meta, dict):
        raise ValueError(f'Invalid frontmatter: {path}')
    return meta, match[2]


def load_pages(root=ROOT, include_candidates=False):
    pages = []
    paths = list((root / 'sources').rglob('*.md'))
    if include_candidates:
        paths.extend((root / 'wiki/candidates').glob('*.md'))
    for path in sorted(paths):
        meta, body = read_page(path)
        pages.append(dict(meta=meta, body=body, path=path.relative_to(root).as_posix()))
    return pages


def captured_bundles(root=ROOT):
    """Map upstream URLs to the latest published local capture with actual code."""
    out = {}
    for path in (root / 'wiki/candidates').glob('*.md'):
        meta, _ = read_page(path)
        artifact = meta.get('artifact_dir')
        if not artifact:
            continue
        bundle = Path(artifact)
        if not bundle.is_absolute():
            bundle = root / bundle
        manifest_path = bundle / 'PROVENANCE.yaml'
        if not manifest_path.exists():
            continue
        manifest = yaml.safe_load(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('errors'):
            continue
        if any(f.get('source_sha') and (bundle / f['path']).is_file() for f in manifest.get('files', [])):
            out[meta['url']] = bundle
    return out


def normalize_arch(value, root=ROOT):
    aliases = yaml.safe_load((root / 'data/aliases.yaml').read_text(encoding='utf-8'))
    needle = value.casefold().strip().replace('_', '').replace('-', '').replace(' ', '')
    for arch, variants in aliases.items():
        if needle in {v.casefold().replace('_', '').replace('-', '').replace(' ', '') for v in [arch, *variants]}:
            return arch
    raise ValueError(f'Unsupported or ambiguous architecture {value!r}; choose SM80/SM86/SM89/SM120.')


def select(pages, architecture=None, kind=None, status=None, repo=None, tags=(), query=''):
    words = query.casefold().split()
    matches = []
    for page in pages:
        m = page['meta']
        if architecture and architecture not in m['architectures']:
            continue
        if kind and kind != m['type']:
            continue
        if status and status != m['status']:
            continue
        if repo and repo.casefold() not in m.get('repo', '').casefold():
            continue
        if not set(tags).issubset(m.get('tags', [])):
            continue
        title = m['title'].casefold()
        metadata = ' '.join([m['id'], *m['tags'], *m['architectures']]).casefold()
        body = page['body'].casefold()
        score = sum(8 * (w in title) + 4 * (w in metadata) + (w in body) for w in words)
        if words and not score:
            continue
        matches.append((score, page))
    matches.sort(key=lambda p: (-p[0], p[1]['meta']['id']))
    return [p for _, p in matches]
