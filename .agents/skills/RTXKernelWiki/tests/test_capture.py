import base64
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from github_capture import capture, safe_path, kernel_path, parse_target, fetch_diff, public_diff
from refresh_candidate_ledger import merge_candidates, select_recipes
from verify_captures import verify_bundle


class FakeGitHub:
    def __init__(self, fail_file=False, moving=False):
        self.fail_file = fail_file
        self.moving = moving
        self.pr_reads = 0
        self.file_refs = []

    def __call__(self, endpoint, accept=None):
        if accept:
            return b'diff --git a/kernel.cu b/kernel.cu\n'
        if endpoint.endswith('/issues/7'):
            return dict(title='SM80 kernel fix', body='Kernel description', state='closed', updated_at='2026-09-17T00:00:00Z', pull_request={})
        if endpoint.endswith('/pulls/7'):
            self.pr_reads += 1
            head = 'a'*40 if not self.moving or self.pr_reads == 1 else 'c'*40
            return dict(title='SM80 kernel fix', state='closed', head=dict(sha=head), base=dict(sha='b'*40), merged_at=None, merge_commit_sha=None, changed_files=3)
        if '/files?' in endpoint:
            return [dict(filename='include/kernel.cuh', status='modified'), dict(filename='csrc/deleted.cu', status='removed'), dict(filename='docs/readme.md', status='modified')]
        if '/contents/' in endpoint:
            if self.fail_file:
                raise RuntimeError('HTTP 404 missing file')
            self.file_refs.append(endpoint)
            data = b'// kernel source\n'
            return dict(type='file', size=len(data), encoding='base64', content=base64.b64encode(data).decode())
        return []


class CaptureTests(unittest.TestCase):
    def test_public_patch_rejects_html_and_oversized_response(self):
        with patch('github_capture.urlopen', return_value=io.BytesIO(b'<html>error</html>')):
            with self.assertRaisesRegex(ValueError, 'unified diff'):
                public_diff('example/kernels', 7)
        with patch('github_capture.urlopen', return_value=io.BytesIO(b'diff --git a b\n' * 10)):
            with self.assertRaisesRegex(ValueError, 'exceeds'):
                public_diff('example/kernels', 7, max_bytes=20)

    def test_large_diff_uses_public_patch_service_and_records_shas(self):
        def api(endpoint, accept=None):
            if accept:
                raise RuntimeError('gh: Sorry, the diff exceeded the maximum number of lines (20000) (HTTP 406)')
            self.fail('Unexpected API fallback')
        with patch('github_capture.public_diff', return_value=b'full patch') as fallback:
            data, provenance = fetch_diff('example/kernels', 7, 'b'*40, 'a'*40, api)
            fallback.assert_called_once_with('example/kernels', 7)
        self.assertEqual(data, b'full patch')
        self.assertEqual(provenance['expected_base_sha'], 'b'*40)
        self.assertEqual(provenance['expected_head_sha'], 'a'*40)

    def test_diff_auth_failure_is_not_retried_as_public(self):
        def api(endpoint, accept=None):
            raise RuntimeError('HTTP 403')
        with patch('github_capture.public_diff') as fallback, self.assertRaisesRegex(RuntimeError, '403'):
            fetch_diff('example/kernels', 7, 'b'*40, 'a'*40, api)
        fallback.assert_not_called()

    def test_fixed_head_and_removed_file_base(self):
        api = FakeGitHub()
        with tempfile.TemporaryDirectory() as directory:
            bundle, result = capture('https://github.com/example/kernels/pull/7', Path(directory), ['sm80'], api=api, write_wiki=False)
            self.assertTrue(result['complete'], result)
            self.assertEqual(result['status'], 'closed-unmerged')
            self.assertTrue(any('deleted.cu?ref=' + 'b'*40 in x for x in api.file_refs))
            self.assertTrue(any('kernel.cuh?ref=' + 'a'*40 in x for x in api.file_refs))
            self.assertTrue((bundle / 'diff.patch').is_file())
            self.assertIn('Kernel description', (bundle / 'page.md').read_text(encoding='utf-8'))
            on_disk = yaml.safe_load((bundle / 'PROVENANCE.yaml').read_text(encoding='utf-8'))
            self.assertEqual(on_disk['head_sha'], 'a'*40)

    def test_failures_do_not_publish_complete_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle, result = capture('https://github.com/example/kernels/pull/7', Path(directory), api=FakeGitHub(fail_file=True), write_wiki=False)
            self.assertFalse(result['complete'])
            self.assertIn('404', result['errors'][0])
            self.assertTrue((bundle / 'PROVENANCE.yaml').is_file())

    def test_moving_pr_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, result = capture('https://github.com/example/kernels/pull/7', Path(directory), api=FakeGitHub(moving=True), write_wiki=False)
            self.assertFalse(result['complete'])
            self.assertIn('changed during capture', result['errors'][0])

    def test_limits_are_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            _, result = capture('https://github.com/example/kernels/pull/7', Path(directory), max_files=1, api=FakeGitHub(), write_wiki=False)
            self.assertFalse(result['complete'])
            self.assertTrue(result['truncations'])

    def test_integrity_check_detects_modified_download(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle, _ = capture('https://github.com/example/kernels/pull/7', Path(directory), api=FakeGitHub(), write_wiki=False)
            self.assertEqual(verify_bundle(bundle), [])
            (bundle / 'key-files/head/include/kernel.cuh').write_text('// changed', encoding='utf-8')
            self.assertTrue(any('hash mismatch' in error for error in verify_bundle(bundle)))

    def test_paths_cannot_escape_or_alias_on_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ('../x', '/tmp/x', 'C:/x', 'a\\..\\x', 'a:b', 'CON.txt', 'x./y'):
                with self.assertRaises(ValueError, msg=relative):
                    safe_path(root, relative)
            self.assertEqual(safe_path(root, 'include/kernel.cuh'), root / 'include/kernel.cuh')

    def test_key_kernel_selection(self):
        for path in ('include/cute/arch/mma_sm80.hpp', 'csrc/gemm.cu', 'examples/python/CuTeDSL/ampere/gemm.py', 'vllm/attention/backends/impl.py', 'src/natten/autotuner.py', 'src/natten/functional.py'):
            self.assertTrue(kernel_path(path), path)
        for path in ('tests/test_gemm.cu', '.github/ci.py', 'docs/tutorial.md', 'setup.py', 'tests/test_natten.py'):
            self.assertFalse(kernel_path(path), path)

    def test_ledger_preserves_triage_and_deduplicates_architectures(self):
        old = [dict(kind='pr', number=7, title='old', architectures=['sm80'], discovery_recipes=['old'], decision='include', reason='manually reviewed')]
        result = dict(recipe=dict(id='new', architecture='sm89'), items=[dict(kind='pr', number=7, title='new title', url='https://github.com/a/b/pull/7', state='closed', updated_at='date')])
        rows = merge_candidates(old, [result, result])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['architectures'], ['sm80','sm89'])
        self.assertEqual(rows[0]['decision'], 'include')
        self.assertEqual(rows[0]['reason'], 'manually reviewed')

    def test_cutoff_is_applied_to_search_not_only_local_filter(self):
        recipes=[dict(id='one',architecture='sm80',kind='pr',query='repo:NVIDIA/cutlass is:pr sm80'), dict(id='two',architecture='sm89',kind='issue',query='repo:NVIDIA/cutlass is:issue sm89')]
        selected = select_recipes(recipes, ['sm80'], ['cutlass'], merged=True, cutoff='2026-09-17')
        self.assertEqual(len(selected), 1)
        self.assertIn('merged:<=2026-09-17', selected[0]['query'])
        self.assertNotIn('created:', recipes[0]['query'])


if __name__ == '__main__':
    unittest.main()
