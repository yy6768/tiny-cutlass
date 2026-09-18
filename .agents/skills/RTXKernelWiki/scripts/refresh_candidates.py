"""Read-only GitHub search through gh; write candidates outside the active corpus."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from urllib.parse import urlencode
from catalog import ROOT


def fetch_recipe(recipe, max_pages, api):
    items = []
    seen = set()
    total = 0
    incomplete = False
    pages = 0
    for page in range(1, max_pages + 1):
        params = urlencode(dict(q=recipe['query'], sort='updated', order='desc', per_page=100, page=page))
        data = api('search/issues?' + params)
        pages += 1
        total = data['total_count']
        incomplete |= data.get('incomplete_results', False)
        for item in data['items']:
            if item['html_url'] in seen:
                continue
            seen.add(item['html_url'])
            items.append({
                'url': item['html_url'], 'number': item['number'], 'title': item['title'],
                'kind': 'pr' if 'pull_request' in item else 'issue',
                'state': item['state'], 'updated_at': item['updated_at'],
                'review': 'unreviewed',
            })
        if len(data['items']) < 100 or len(items) >= min(total, 1000):
            break
    return dict(recipe=recipe, total_count=total, fetched_pages=pages, incomplete_results=incomplete,
                truncated=len(items) < total, items=items)


def github_api(endpoint):
    p = subprocess.run(['gh', 'api', endpoint], capture_output=True, encoding='utf-8', timeout=60)
    if p.returncode:
        raise RuntimeError(p.stderr.strip())
    return json.loads(p.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--recipe', action='append', default=[])
    parser.add_argument('--max-pages', type=int, default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    recipes = json.loads((ROOT / 'queries/search-recipes.json').read_text(encoding='utf-8'))
    if args.list:
        for r in recipes:
            print(f"{r['id']}: {r['query']}")
        return
    if not 1 <= args.max_pages <= 10:
        parser.error('--max-pages must be 1..10 (GitHub Search limit: 1000 results)')
    if not args.output or not args.recipe:
        parser.error('Specify --recipe and --output; use --list for recipe IDs.')
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error('--output must be outside the skill; candidates require review before inclusion.')
    selected = [r for r in recipes if r['id'] in args.recipe]
    missing = set(args.recipe) - {r['id'] for r in selected}
    if missing:
        parser.error(f'Unknown recipes: {sorted(missing)}')
    result = dict(retrieved_at=datetime.now(timezone.utc).isoformat(), results=[], errors=[])
    for r in selected:
        try:
            result['results'].append(fetch_recipe(r, args.max_pages, github_api))
        except (OSError, RuntimeError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
            result['errors'].append(dict(recipe=r['id'], error=str(error)))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Saved {len(result['results'])} query results and {len(result['errors'])} errors to {output}")
    if result['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
