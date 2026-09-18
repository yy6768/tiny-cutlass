"""Validate source provenance, architecture scope, links, and generated indices."""
from datetime import date
import json
import re
from urllib.parse import unquote, urlsplit
from catalog import ROOT, ARCHITECTURES, TYPES, STATUSES, load_pages
from generate_queries import rendered_indices


def validate(root=ROOT):
    errors = []
    try:
        pages = load_pages(root)
    except (ValueError, OSError) as error:
        return [str(error)]
    ids = set()
    counts = {arch: {'pr': 0, 'issue': 0, 'blog': 0} for arch in ARCHITECTURES}
    required = {'id', 'title', 'type', 'url', 'architectures', 'architecture_evidence', 'tags',
                'kernel_types', 'symptoms', 'status', 'confidence', 'retrieved_at',
                'inclusion_reason', 'local_validation'}
    for p in pages:
        m = p['meta']
        path = p['path']
        def check(condition, message):
            if not condition:
                errors.append(f'{path}: {message}')
        check(required <= m.keys(), f'missing fields {required - m.keys()}')
        if not required <= m.keys():
            continue
        check(m['id'] not in ids, 'duplicate ID')
        ids.add(m['id'])
        check(m['type'] in TYPES, 'invalid type')
        check(m['status'] in STATUSES, 'invalid status')
        check(bool(m['architectures']) and set(m['architectures']) <= set(ARCHITECTURES), 'out-of-scope architecture')
        check(set(m['architecture_evidence']) == set(m['architectures']), 'architecture evidence mismatch')
        check(all(m['architecture_evidence'].values()), 'empty architecture evidence')
        check(m['confidence'] in ('source-reported', 'inferred'), 'unsupported confidence claim')
        check(m['local_validation'] == 'not-run', 'this corpus has no local GPU validation artifacts')
        check(m['url'].startswith('https://'), 'expected upstream HTTPS URL')
        try:
            date.fromisoformat(m['retrieved_at'])
        except (TypeError, ValueError):
            check(False, 'retrieved_at must be a quoted ISO date')
        for key in ('architectures', 'tags', 'kernel_types', 'symptoms'):
            check(isinstance(m[key], list), f'{key} must be a list')
        for arch in m['architectures']:
            if arch in counts and m['type'] in counts[arch]:
                counts[arch][m['type']] += 1
        if m['type'] in ('pr', 'issue'):
            check(all(m.get(k) for k in ('repo', 'number', 'updated_at')), 'missing GitHub identity/date')
            route = 'pull' if m['type'] == 'pr' else 'issues'
            check(m['url'] == f"https://github.com/{m.get('repo')}/{route}/{m.get('number')}", 'GitHub URL/type mismatch')
        if m['type'] == 'pr':
            check(m['status'] in ('merged', 'open', 'closed-unmerged'), 'invalid PR state')
            check(bool(m.get('merged_at')) == (m['status'] == 'merged'), 'closed is not merged')
            check(bool(re.fullmatch('[0-9a-f]{40}', m.get('head_sha', ''))), 'missing immutable PR head SHA')
            check(bool(m.get('changed_files')) and bool(m.get('code_urls')), 'missing changed files/code links')
            check(all(m['head_sha'] in u for u in m.get('code_urls', [])), 'code URL is not pinned to PR head')
        elif m['type'] == 'issue':
            check(m['status'] in ('open', 'closed'), 'issue cannot be merged')
        else:
            check(m['status'] == 'published', 'doc/blog must be published')
    for arch, kinds in counts.items():
        for kind, n in kinds.items():
            if not n:
                errors.append(f'Missing curated {kind} coverage for {arch}')
    # Raw upstream snapshots retain upstream-relative links; only maintained pages are checked here.
    maintained = [root / 'SKILL.md']
    for folder in ('sources', 'references', 'queries', 'candidates', 'wiki'):
        maintained.extend((root / folder).rglob('*.md'))
    for path in maintained:
        for link in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if urlsplit(link).scheme or link.startswith('#'):
                continue
            target = (path.parent / unquote(link.split('#')[0])).resolve()
            if not target.exists():
                errors.append(f'{path.relative_to(root)}: broken link {link}')
    recipes = json.loads((root / 'queries/search-recipes.json').read_text(encoding='utf-8'))
    recipe_ids = set()
    for r in recipes:
        if r['id'] in recipe_ids or r['architecture'] not in ARCHITECTURES or f"is:{r['kind']}" not in r['query']:
            errors.append(f'Invalid recipe {r["id"]}')
        recipe_ids.add(r['id'])
    for name, text in rendered_indices(root).items():
        path = root / 'queries' / name
        if not path.exists() or path.read_text(encoding='utf-8') != text:
            errors.append(f'Stale generated index {name}')
    return errors


if __name__ == '__main__':
    errors = validate()
    if errors:
        print('\n'.join(errors))
        raise SystemExit(1)
    print(f'Validated {len(load_pages())} sources, architecture coverage, provenance, local links and indices.')
