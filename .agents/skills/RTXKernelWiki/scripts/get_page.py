"""Read a curated page by exact ID or corpus-relative path."""
import argparse
from catalog import ROOT, load_pages, captured_bundles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('page')
    parser.add_argument('--body-only', action='store_true')
    parser.add_argument('--include-code', action='store_true', help='Show captured key kernel files for a candidate page')
    args = parser.parse_args()
    needle = args.page.replace('\\', '/')
    pages = [p for p in load_pages(include_candidates=True) if needle in (p['meta']['id'], p['path'])]
    if len(pages) != 1:
        parser.error('Expected one curated page; use query.py to find its ID.')
    p = pages[0]
    print(p['body'] if args.body_only else (ROOT / p['path']).read_text(encoding='utf-8'))
    if args.include_code:
        from pathlib import Path
        import yaml
        from github_capture import safe_path
        artifact = p['meta'].get('artifact_dir')
        if not artifact:
            artifact = captured_bundles().get(p['meta']['url'])
        if not artifact:
            parser.error('No local capture. Run fetch_pr_diff.py for this source first.')
        bundle = Path(artifact)
        if not bundle.is_absolute():
            bundle = ROOT / bundle
        manifest = yaml.safe_load((bundle / 'PROVENANCE.yaml').read_text(encoding='utf-8'))
        for file in manifest['files']:
            if 'source_sha' in file:
                print(f"\n--- {file['upstream_path']} @ {file['source_sha']} ---\n")
                print(safe_path(bundle, file['path']).read_text(encoding='utf-8', errors='replace'))


if __name__ == '__main__':
    main()
