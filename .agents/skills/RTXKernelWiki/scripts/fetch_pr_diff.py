"""Fetch candidate PR/issue pages, discussion, diffs and pinned kernel files."""
import argparse
from pathlib import Path
import yaml
from catalog import normalize_arch
from github_capture import capture, parse_target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('targets', nargs='*', help='Curated source IDs or GitHub PR/issue URLs')
    parser.add_argument('--ids', nargs='+', default=[], help='Original ID-based capture entry point')
    parser.add_argument('--ledger', type=Path, help='Candidate YAML emitted by refresh_candidate_ledger.py')
    parser.add_argument('--decision', nargs='+', choices=['include','defer','exclude'], default=['include'])
    parser.add_argument('--architecture', action='append', default=[])
    parser.add_argument('--output', type=Path, help='Default: skill artifacts/github')
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--max-files', type=int, default=40)
    parser.add_argument('--max-file-bytes', type=int, default=1024*1024)
    parser.add_argument('--max-bundle-bytes', type=int, default=20*1024*1024)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if min(args.limit, args.max_files, args.max_file_bytes, args.max_bundle_bytes) < 1:
        parser.error('Limits must be positive')
    try:
        archs = [normalize_arch(x) for x in args.architecture]
        jobs = {x: list(archs) for x in args.targets + args.ids}
        if args.ledger:
            ledger = yaml.safe_load(args.ledger.read_text(encoding='utf-8'))
            for row in ledger['candidates']:
                if row['decision'] in args.decision and (not archs or set(archs) & set(row['architectures'])):
                    jobs[row['url']] = row['architectures']
        if not jobs:
            parser.error('Supply URLs/IDs or a ledger with matching candidates.')
        # Validate every input before beginning a network capture.
        for target in jobs:
            parse_target(target)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    errors = 0
    for target, archs in list(jobs.items())[:args.limit]:
        if args.dry_run:
            print(f'{target} architectures={archs}')
            continue
        bundle, result = capture(target, args.output, archs, args.max_files, args.max_file_bytes,
                                 max_bundle_bytes=args.max_bundle_bytes)
        print(f'{target} -> {bundle} ({"complete" if result["complete"] else "incomplete"})')
        for error in result['errors']:
            print(f'  ERROR: {error}')
        for reason in result['truncations']:
            print(f'  TRUNCATED: {reason}')
        for skipped in result['skipped']:
            print(f'  SKIPPED: {skipped["upstream_path"]}: {skipped["reason"]}')
        errors += not result['complete']
    if len(jobs) > args.limit:
        print(f'{len(jobs) - args.limit} targets not fetched due to --limit={args.limit}.')
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
