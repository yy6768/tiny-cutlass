"""Search the curated SM80/86/89/120 corpus; never search candidate caches."""
import argparse
import json
from catalog import ARCHITECTURES, TYPES, STATUSES, load_pages, normalize_arch, select, captured_bundles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('query', nargs='*', help='Keywords (OR ranking, including Chinese terms in summaries)')
    parser.add_argument('--architecture', help='/'.join(ARCHITECTURES) + ' or a documented GPU alias')
    parser.add_argument('--type', choices=TYPES)
    parser.add_argument('--status', choices=STATUSES)
    parser.add_argument('--repo')
    parser.add_argument('--tag', action='append', default=[], help='Repeat for AND filtering')
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--paths-only', action='store_true')
    parser.add_argument('--compact', action='store_true')
    parser.add_argument('--include-candidates', action='store_true', help='Include downloaded wiki candidate pages, labelled unreviewed')
    parser.add_argument('--has-code', action='store_true', help='Only pages with a local capture containing key source files')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    try:
        arch = normalize_arch(args.architecture) if args.architecture else None
    except ValueError as error:
        parser.error(str(error))
    pages = select(load_pages(include_candidates=args.include_candidates), arch, args.type, args.status, args.repo, args.tag, ' '.join(args.query))
    if args.has_code:
        captures = captured_bundles()
        pages = [p for p in pages if p['meta']['url'] in captures]
    pages = pages[:args.limit]
    if args.json:
        print(json.dumps([dict(path=p['path'], **p['meta']) for p in pages], ensure_ascii=False, indent=2))
    elif args.paths_only:
        print('\n'.join(p['path'] for p in pages))
    else:
        for p in pages:
            m = p['meta']
            review = '; candidate/unreviewed' if m.get('review') == 'defer' else ''
            print(f"{m['id']} [{m['status']}; {','.join(m['architectures'])}{review}] {m['title']}")
            if not args.compact:
                print(f"  {p['path']}\n  {m['url']}\n  {m['inclusion_reason']}")
        print(f'{len(pages)} result(s); architecture tags indicate relevance, not local GPU validation.')


if __name__ == '__main__':
    main()
