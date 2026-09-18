"""Discover architecture-specific PR/issues, preserve triage, and optionally capture implementations."""
import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path
import re
import time
import yaml
from catalog import ROOT, normalize_arch
from refresh_candidates import fetch_recipe, github_api
from github_capture import capture


def merge_candidates(existing, results):
    """Refresh discoveries without replacing the user's include/defer/exclude decisions."""
    rows = {(r['kind'], r['number']): dict(r) for r in existing}
    for result in results:
        recipe = result['recipe']
        for item in result['items']:
            key = (item['kind'], item['number'])
            row = rows.setdefault(key, dict(kind=item['kind'], number=item['number'], decision='defer',
                                           reason='Architecture search hit; inspect implementation before inclusion.',
                                           architectures=[], discovery_recipes=[]))
            row.update({k: item[k] for k in ('url', 'title', 'state', 'updated_at')})
            row['architectures'] = sorted(set(row['architectures']) | {recipe['architecture']})
            row['discovery_recipes'] = sorted(set(row['discovery_recipes']) | {recipe['id']})
    return sorted(rows.values(), key=lambda r: (r['kind'], r['number']))


def select_recipes(recipes, architectures, repos=(), kind='both', ids=(), since=None, cutoff=None, merged=False):
    selected = []
    for source in recipes:
        repo = re.search(r'\brepo:([^ ]+)', source['query']).group(1)
        if architectures and source['architecture'] not in architectures:
            continue
        if repos and repo.casefold() not in {r.casefold() for r in repos} and repo.split('/')[-1].casefold() not in {r.casefold() for r in repos}:
            continue
        if kind != 'both' and source['kind'] != kind:
            continue
        if ids and source['id'] not in ids:
            continue
        recipe = dict(source, repo=repo)
        if since:
            recipe['query'] += f' updated:>={since}'
        if cutoff:
            recipe['query'] += f' created:<={cutoff}'
        if merged and recipe['kind'] == 'pr':
            recipe['query'] += ' is:merged'
            if cutoff:
                recipe['query'] += f' merged:<={cutoff}'
        selected.append(recipe)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture', '--arch', action='append', default=[])
    parser.add_argument('--repos', help='Comma-separated slugs or owner/repo names')
    parser.add_argument('--kind', choices=['pr', 'issue', 'both'], default='both')
    parser.add_argument('--recipe', action='append', default=[])
    parser.add_argument('--since', type=date.fromisoformat, help='GitHub updated: lower bound')
    parser.add_argument('--cutoff', type=date.fromisoformat, help='Creation/merge upper bound, not historical state reconstruction')
    parser.add_argument('--merged', action='store_true', help='Only merged PRs; issue queries unchanged')
    parser.add_argument('--max-pages', type=int, default=1)
    parser.add_argument('--delay', type=float, default=2.1, help='Minimum seconds between GitHub search requests')
    parser.add_argument('--output', type=Path, default=ROOT / 'candidates/github', help='Ledger directory')
    parser.add_argument('--fetch', action='store_true', help='Also download pages, diffs and key kernel files for discovered candidates')
    parser.add_argument('--fetch-limit', type=int, default=10)
    parser.add_argument('--dry-run', action='store_true', help='Print search plan without network or writes')
    args = parser.parse_args()
    if not 1 <= args.max_pages <= 10 or args.fetch_limit < 1 or args.delay < 0:
        parser.error('max-pages=1..10, fetch-limit>0, delay>=0 required')
    recipes = json.loads((ROOT / 'queries/search-recipes.json').read_text(encoding='utf-8'))
    try:
        archs = [normalize_arch(a) for a in args.architecture]
    except ValueError as error:
        parser.error(str(error))
    if set(args.recipe) - {r['id'] for r in recipes}:
        parser.error('Unknown recipe ID')
    repos = [r.strip() for r in (args.repos or '').split(',') if r.strip()]
    selected = select_recipes(recipes, archs, repos, args.kind, args.recipe, args.since, args.cutoff, args.merged)
    if not selected:
        parser.error('No recipes match these filters')
    if args.dry_run:
        for recipe in selected:
            print(f'{recipe["id"]}: {recipe["query"]}')
        return
    now = datetime.now(timezone.utc).isoformat()
    report = dict(retrieved_at=now, results=[], errors=[])
    last = 0.0
    def api(endpoint):
        nonlocal last
        time.sleep(max(0, args.delay - (time.monotonic() - last)))
        last = time.monotonic()
        return github_api(endpoint)
    for i, recipe in enumerate(selected, 1):
        print(f'[{i}/{len(selected)}] {recipe["id"]}', flush=True)
        try:
            result = fetch_recipe(recipe, args.max_pages, api)
            report['results'].append(result)
            print(f'  {len(result["items"])}/{result["total_count"]} hits, truncated={result["truncated"]}', flush=True)
        except Exception as error:
            report['errors'].append(dict(recipe=recipe['id'], error=str(error)))
            print(f'  ERROR: {error}', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    found = {}
    for repo in sorted({r['repo'] for r in selected}):
        results = [r for r in report['results'] if r['recipe']['repo'] == repo]
        if not results:
            continue
        file = args.output / f'{repo.replace("/", "--")}.yaml'
        ledger = yaml.safe_load(file.read_text(encoding='utf-8')) if file.exists() else dict(repo=repo, candidates=[])
        if ledger['repo'] != repo:
            raise ValueError(f'Unexpected repo in {file}')
        ledger['candidates'] = merge_candidates(ledger['candidates'], results)
        ledger['last_attempt_at'] = now
        ledger['last_search_partial'] = (any(r['truncated'] or r['incomplete_results'] for r in results)
                                         or len(results) != sum(r['repo'] == repo for r in selected))
        ledger['last_queries'] = [r['recipe'] for r in results]
        file.write_text(yaml.safe_dump(ledger, allow_unicode=True, sort_keys=False), encoding='utf-8')
        print(f'Ledger: {file}', flush=True)
        hit_urls = {i['url'] for r in results for i in r['items']}
        for row in ledger['candidates']:
            if row['url'] in hit_urls and row['decision'] != 'exclude':
                found[row['url']] = row
    report['captures'] = []
    if args.fetch:
        for url, row in list(found.items())[:args.fetch_limit]:
            bundle, result = capture(url, architectures=row['architectures'])
            report['captures'].append(dict(url=url, bundle=str(bundle), complete=result['complete'], errors=result['errors']))
            print(f'Capture: {url} -> {bundle}', flush=True)
        report['capture_limit_skipped'] = max(0, len(found) - args.fetch_limit)
        if report['capture_limit_skipped']:
            print(f'{report["capture_limit_skipped"]} candidates remain; increase --fetch-limit or use fetch_pr_diff.py --ledger.')
    (args.output / 'search-results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if report['errors'] or any(not c['complete'] for c in report['captures']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
