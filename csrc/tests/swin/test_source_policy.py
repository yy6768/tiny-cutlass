"""CPU-only policy and real Windows entrypoint regression tests.

Run: python -B -m unittest discover -s csrc/tests/swin -p test_source_policy.py -v
All fixture workspaces and logs stay under build/.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'csrc/tests/common'))
from kernel_policy import RULES, check_policy, scan_source


class PolicyChecks(unittest.TestCase):
    def setUp(self):
        parent = ROOT / 'build/kda-local-maintenance/tests'
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='policy ', dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.write('csrc/swin/window_attention/warp/mma.h', '#include "cutlass/gemm/warp/default_mma_tensor_op.h"\n')
        self.config = json.loads((ROOT / 'csrc/swin/kernel-policy.json').read_text())
        self.policy = self.root / 'csrc/swin/kernel-policy.json'

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        return path

    def check(self):
        self.policy.write_text(json.dumps(self.config), encoding='utf-8')
        return check_policy(self.root, self.policy)

    def copy_runtime(self):
        for name in ('csrc/tests/common/kernel_policy.py', 'csrc/tests/swin/source_policy.py',
                     'csrc/tests/swin/verify.py', 'csrc/tests/swin/bench.py'):
            self.write(name, (ROOT / name).read_text(encoding='utf-8'))
        self.check()

    def invoke(self, script, *args):
        return subprocess.run([sys.executable, '-S', '-B', str(self.root / script), *args],
                              cwd=self.root, capture_output=True, text=True)

    def test_stock_tensorop_and_cuda_glue(self):
        self.source.write_text('''
#include <cutlass/gemm/warp/default_mma_tensor_op.h>
using Policy = DefaultWindowAttention<cutlass::arch::Sm89, cutlass::half_t>;
__device__ float norm(float x) { return __shfl_xor_sync(0xffffffff, rsqrtf(x), 1); }
cutlass::Status unsupported = cutlass::Status::kErrorNotSupported;
''')
        self.assertEqual(self.check().diagnostics, [])

    def test_rules_detect_dependencies_and_unqualified_types(self):
        samples = {
            'inline-ptx': ['asm volatile("mma.sync;" : :);', '__asm__("bar.sync 0;");'],
            'cute': ['#include <cute/tensor.hpp>', 'namespace ct = cute;', 'using namespace cute;', 'cute::make_tensor(ptr);'],
            'cutlass3': ['#include "cutlass/gemm/collective/collective_builder.hpp"', 'using Mma = CollectiveMma<X>;'],
            'torch-core': ['#include <ATen/ATen.h>', 'at::Tensor tensor;', '#include <torch/extension.h>', 'c10::Device device;', 'namespace framework = torch;'],
            'simt-mma': ['#include <cutlass/gemm/warp/mma_simt.h>', 'using Op = cutlass::arch::OpClassSimt;', 'MmaSimt<X> mma;'],
        }
        for rule, sources in samples.items():
            for source in sources:
                with self.subTest(rule=rule, source=source):
                    self.source.write_text('\n' + source)
                    findings = self.check().diagnostics
                    self.assertEqual([(f.line, f.rule) for f in findings], [(2, rule)])

    def test_comments_literals_and_line_continuations(self):
        source = '''// cute::foo() and asm("mma.sync;");
/* #include <torch/extension.h>
   OpClassSimt */
auto text = "https://site/* asm cute:: at:: */";
auto raw = R"doc(
#include <cute/tensor.hpp>
asm volatile("mma.sync;");
)doc";
char quote = '\\'';
// continued comment \\
asm("this is still a comment");
#include \\
  <ATen/ATen.h>
'''
        findings = scan_source('example.cu', source, set(RULES))
        self.assertEqual([(f.line, f.rule) for f in findings], [(12, 'torch-core')])
        findings = scan_source('digits.cu', "int count = 1'000;\nasm(\"mma.sync;\");\nchar c = 'x';", set(RULES))
        self.assertEqual([(f.line, f.rule) for f in findings], [(2, 'inline-ptx')])

    def test_scope_excludes_vendor_and_generated_artifacts(self):
        self.write('3rdparty/cutlass/include/cutlass/arch/mma.h', 'asm("mma.sync;");')
        self.write('build/swin/kernel.cu', 'cute::make_tensor();')
        self.write('csrc/swin/window_attention/third_party/example.cu', 'at::Tensor x;')
        self.write('csrc/swin/window_attention/generated.ptx', 'asm("mma.sync;");')
        self.write('csrc/swin/patch_embed/example.cu', 'cute::make_tensor();')
        report = self.check()
        self.assertEqual((report.files, report.diagnostics), (1, []))
        self.write('csrc/swin/window_attention/builders/builder.h', 'cutlass::arch::OpClassSimt x;')
        report = self.check()
        self.assertEqual(report.files, 2)
        self.assertEqual([d.rule for d in report.diagnostics], ['simt-mma'])

    def exception(self, rule='inline-ptx'):
        decision = 'csrc/swin/docs/exception.md'
        self.write(decision, 'Primitive interface and missing library capability; candidate awaiting parity and timing.')
        return {'path': self.source.relative_to(self.root).as_posix(), 'rule': rule,
                'reason': 'Isolated primitive candidate for a documented library gap.', 'decision': decision}

    def test_exception_is_visible_and_limited_to_one_file_and_rule(self):
        self.source.write_text('asm("mma.sync;");')
        self.config['exceptions'] = [self.exception()]
        report = self.check()
        self.assertEqual(len(report.exceptions), 1)
        self.assertEqual(report.diagnostics, [])
        self.write('csrc/swin/window_attention/warp/other.h', 'asm("mma.sync;");')
        self.source.write_text('asm("mma.sync;");\nat::Tensor x;')
        findings = self.check().diagnostics
        self.assertEqual({f.rule for f in findings}, {'inline-ptx', 'torch-core'})
        self.assertEqual(len(findings), 2)

    def test_invalid_exceptions_fail_closed(self):
        self.source.write_text('asm("mma.sync;");')
        good = self.exception()
        cases = [
            [{**good, 'rule': 'torch-core'}], [{**good, 'rule': 'simt-mma'}],
            [{**good, 'rule': 'unknown'}], [{**good, 'path': 'csrc/swin/window_attention/*'}],
            [{**good, 'path': '../outside.h'}], [{**good, 'decision': 'missing.md'}],
            [{**good, 'reason': ''}], [good, good], [{**good, 'rule': 'cute'}],
            [{**good, 'path': 'csrc/swin/window_attention/warp/missing.h'}],
        ]
        for exceptions in cases:
            with self.subTest(exceptions=exceptions):
                self.config['exceptions'] = exceptions
                self.assertIn('configuration', {d.rule for d in self.check().diagnostics})

    def test_missing_scope_and_malformed_config_fail_closed(self):
        self.config['source_roots'] = ['csrc/absent']
        self.assertEqual(self.check().diagnostics[0].rule, 'configuration')
        self.policy.write_text('{broken')
        self.assertEqual(check_policy(self.root, self.policy).diagnostics[0].rule, 'configuration')

    def test_source_only_needs_no_site_packages(self):
        self.copy_runtime()
        result = self.invoke('csrc/tests/swin/verify.py', '--source-only')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('SOURCE POLICY PASSED', result.stdout)

    def test_direct_entrypoints_stop_before_torch_or_gpu_on_violation(self):
        self.source.write_text('asm("mma.sync;");')
        self.copy_runtime()
        for script, modes in (
                ('verify.py', [[], ['--source-only'], ['--block'], ['--case', 'swin_grouped_gemm']]),
                ('bench.py', [[], ['--block'], ['--grouped-gemm']])):
            for mode in modes:
                with self.subTest(script=script, mode=mode):
                    result = self.invoke('csrc/tests/swin/' + script, *mode)
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertIn('[inline-ptx]', result.stderr)
                    self.assertNotIn('ModuleNotFoundError', result.stderr)

    @unittest.skipUnless(sys.platform == 'win32', 'actual cmd.exe orchestration')
    def test_batch_routes_and_failure_order(self):
        # Run family routing/errorlevel checks with CPU substitutes for the
        # external programs. The copied batch adds CALL for the CMake .bat stub.
        self.write('scripts/kernels/swin/swin.bat', (ROOT / 'scripts/kernels/swin/swin.bat').read_text())
        self.write('scripts/kernels/swin/run.bat', (ROOT / 'scripts/kernels/swin/run.bat').read_text())
        stub = '''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
kind = "source" if "--source-only" in args else Path(__file__).stem
with open(os.environ["POLICY_TRACE"], "a") as f: f.write(json.dumps([kind, args]) + "\\n")
sys.exit(17 if os.environ.get("POLICY_FAIL") == kind else 0)
'''
        for name in ('verify', 'bench'):
            self.write(f'csrc/tests/swin/{name}.py', stub)
        self.write('stubs/cmake.py', stub.replace('Path(__file__).stem', '("build" if "--build" in args else "configure")'))
        cmake_stub = f'@echo off\r\n"{sys.executable}" -B "%~dp0cmake.py" %*\r\nexit /b %errorlevel%\r\n'
        self.write('stubs/cmake.bat', cmake_stub)
        # Unlike cmake.exe, a .bat stub needs CALL to return to its caller.
        family = self.root / 'scripts/kernels/swin/swin.bat'
        family.write_text(family.read_text().replace('\ncmake ', '\ncall cmake '))
        trace = self.root / 'trace.jsonl'
        env = {k: v for k, v in os.environ.items() if k.upper() != 'PATH'}
        env.update({'PATH': str(self.root / 'stubs') + os.pathsep + os.environ.get('PATH', ''),
                    'PYTHON': sys.executable, 'POLICY_TRACE': str(trace), 'CUTLASS_PROFILE': '0'})
        routes = [('', 'swin', ['--all-families'], []),
                  ('block', 'swin_block', ['--block'], ['--block']),
                  ('grouped-gemm', 'swin_grouped_gemm', ['--case', 'swin_grouped_gemm'], ['--grouped-gemm'])]
        for mode, target, verify_flags, bench_flags in routes:
            for fail, expected in (('', ['source', 'configure', 'build', 'verify', 'bench']),
                                   ('source', ['source']), ('build', ['source', 'configure', 'build']),
                                   ('verify', ['source', 'configure', 'build', 'verify'])):
                with self.subTest(mode=mode, fail=fail):
                    trace.write_text('')
                    env['POLICY_FAIL'] = fail
                    result = subprocess.run(['cmd.exe', '/d', '/c', 'scripts\\kernels\\swin\\run.bat', mode],
                                            cwd=self.root, env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 1 if fail else 0, result.stdout + result.stderr)
                    calls = [json.loads(line) for line in trace.read_text().splitlines()]
                    self.assertEqual([c[0] for c in calls], expected)
                    if not fail:
                        self.assertEqual(calls[2][1][calls[2][1].index('--target') + 1], target)
                        self.assertEqual(calls[3][1][2:], verify_flags)
                        self.assertEqual(calls[4][1][2:], bench_flags)


if __name__ == '__main__':
    unittest.main()
