import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from catalog import load_pages, normalize_arch, select
from refresh_candidates import fetch_recipe
from validate import validate


class CatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pages = load_pages()

    def test_arch_aliases_do_not_merge_architectures(self):
        for value, arch in [('sm_80','sm80'), ('RTX 3090','sm86'), ('RTX 4090','sm89'), ('sm_120a','sm120')]:
            self.assertEqual(normalize_arch(value), arch)
        for value in ('Blackwell', 'B200', 'H100', 'DGX Spark', 'sm121'):
            with self.assertRaises(ValueError):
                normalize_arch(value)

    def test_architecture_and_type_filters(self):
        for arch in ('sm80', 'sm86', 'sm89', 'sm120'):
            for kind in ('pr', 'issue', 'blog'):
                found = select(self.pages, architecture=arch, kind=kind)
                self.assertTrue(found)
                self.assertTrue(all(arch in p['meta']['architectures'] and p['meta']['type'] == kind for p in found))

    def test_closed_pr_is_not_merged(self):
        merged = select(self.pages, kind='pr', status='merged')
        self.assertTrue(merged)
        self.assertTrue(all(p['meta']['merged_at'] for p in merged))
        closed = select(self.pages, kind='pr', status='closed-unmerged')
        self.assertTrue(closed)
        self.assertTrue(all(not p['meta']['merged_at'] for p in closed))

    def test_withdrawn_claims_not_in_search_corpus(self):
        ids = {p['meta']['id'] for p in self.pages}
        self.assertNotIn('pr-cutlass-3120', ids)
        self.assertNotIn('pr-triton-9852', ids)

    def test_cwd_independent_query(self):
        p = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/query.py'), '--architecture', 'RTX 4090', '--type', 'pr', '--status', 'merged', '--json'], cwd=ROOT.parent, capture_output=True, encoding='utf-8')
        self.assertEqual(p.returncode, 0, p.stderr)
        data = json.loads(p.stdout)
        self.assertTrue(data)
        self.assertTrue(all(x['type'] == 'pr' and x['status'] == 'merged' and 'sm89' in x['architectures'] for x in data))

    def test_windows_style_path_and_issue_lookup(self):
        p = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/get_page.py'), 'sources\\issues\\cutlass-1181.md', '--body-only'], capture_output=True, encoding='utf-8')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('ArchTag', p.stdout)

    def test_refresh_reports_truncation_without_promoting(self):
        items = [dict(html_url=f'https://github.com/a/b/issues/{n}', number=n, title='test', state='closed', updated_at='2026-09-17T00:00:00Z') for n in range(100)]
        r = fetch_recipe(dict(query='repo:a/b is:issue sm120'), 1, lambda _: dict(total_count=200, incomplete_results=True, items=items))
        self.assertTrue(r['truncated'])
        self.assertTrue(r['incomplete_results'])
        self.assertEqual(r['fetched_pages'], 1)
        self.assertTrue(all(x['review'] == 'unreviewed' for x in r['items']))

    def test_refresh_paginates_and_deduplicates(self):
        def api(endpoint):
            start = 0 if endpoint.endswith('page=1') else 99
            stop = 100 if start == 0 else 102
            return dict(total_count=102, incomplete_results=False, items=[dict(html_url=f'https://github.com/a/b/issues/{n}', number=n, title='test', state='open', updated_at='2026-09-17T00:00:00Z') for n in range(start, stop)])
        r = fetch_recipe(dict(query='repo:a/b is:issue sm80'), 2, api)
        self.assertEqual(len(r['items']), 102)
        self.assertFalse(r['truncated'])

    def test_indices_and_provenance(self):
        self.assertEqual(validate(), [])


if __name__ == '__main__':
    unittest.main()
