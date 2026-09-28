"""Source-level kernel policy checks. Standard library only; not a C++ compiler.

Checks local source text, including inactive preprocessor branches. Library
internals, macro expansion, data flow and semantic fallback detection require
source review and target-device validation.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
import sys


RULES = frozenset({'inline-ptx', 'cute', 'cutlass3', 'torch-core', 'simt-mma'})
EXCEPTION_RULES = frozenset({'inline-ptx', 'cute', 'cutlass3'})
SOURCE_SUFFIXES = frozenset({'.h', '.hpp', '.cuh', '.cu', '.cpp', '.cc', '.c'})
EXCLUDED_DIRS = frozenset({'3rdparty', 'third_party', 'build', 'profile', '.git', '__pycache__'})

# Strings must be recognized before searching their contents for comment markers.
LEXEMES = re.compile(
    r'(?P<comment>//[^\n]*|/\*.*?\*/)'
    r'|(?P<raw>(?:u8|u|U|L)?R"(?P<delimiter>[^\s()\\]{0,16})\(.*?\)(?P=delimiter)")'
    r'|(?P<string>"(?:\\.|[^"\\])*"|(?<!\w)(?:u8|u|U|L)?\'(?:\\.|[^\'\\\n])*\')',
    re.DOTALL,
)
INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*[<"]([^>"\n]+)[>"]', re.MULTILINE)
PATTERNS = {
    'inline-ptx': re.compile(r'\b(?:asm|__asm|__asm__)\b'),
    'cute': re.compile(r'\bcute\s*::|\busing\s+namespace\s+cute\b|\bnamespace\s+\w+\s*=\s*cute\b'),
    'cutlass3': re.compile(r'\b(?:CollectiveBuilder|CollectiveMma|CollectiveEpilogue|GemmUniversalAdapter)\b|\bcutlass\s*::\s*(?:gemm|epilogue)\s*::\s*collective\b'),
    'torch-core': re.compile(r'\b(?:torch|at|c10)\s*::|\busing\s+namespace\s+(?:torch|at|c10)\b|\bnamespace\s+\w+\s*=\s*(?:torch|at|c10)\b'),
    'simt-mma': re.compile(r'\b(?:OpClassSimt|MmaSimt|MmaSimtPolicy|DefaultMmaSimt)\b'),
}


@dataclass(frozen=True)
class Diagnostic:
    path: str
    line: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f'{self.path}:{self.line}: [{self.rule}] {self.message}'


@dataclass
class Report:
    files: int
    diagnostics: list[Diagnostic]
    exceptions: list[Diagnostic]


def _views(text: str) -> tuple[str, str, list[int]]:
    """Join C++ line continuations, then mask literals/comments, retaining lines."""
    parts, lines = [], []
    for number, line in enumerate(text.splitlines(keepends=True), 1):
        piece = re.sub(r'\\\r?\n$', '', line)
        parts.append(piece)
        lines.extend([number] * len(piece))
    source = ''.join(parts)
    comments, code = list(source), list(source)
    for match in LEXEMES.finditer(source):
        for index in range(match.start(), match.end()):
            if source[index] != '\n':
                code[index] = ' '
                if match.group('comment') is not None:
                    comments[index] = ' '
    return ''.join(comments), ''.join(code), lines


def scan_source(path: str, text: str, rules: set[str]) -> list[Diagnostic]:
    comments, code, lines = _views(text)
    found: dict[tuple[int, str], Diagnostic] = {}

    def add(rule: str, offset: int, message: str) -> None:
        if rule in rules:
            line = lines[offset]
            found[line, rule] = Diagnostic(path, line, rule, message)

    for match in INCLUDE.finditer(comments):
        # A directive inside a raw string has been masked from the code view.
        if '#' not in code[match.start():match.start(1)]:
            continue
        header = match.group(1).replace('\\', '/')
        if header.startswith('cute/'):
            add('cute', match.start(), f'direct CuTe dependency: {header}')
        if (header.startswith(('cutlass/gemm/collective/', 'cutlass/epilogue/collective/', 'cutlass/pipeline/'))
                or header in {'cutlass/gemm/kernel/gemm_universal.hpp', 'cutlass/gemm/device/gemm_universal_adapter.h'}):
            add('cutlass3', match.start(), f'CUTLASS 3.x dependency: {header}')
        if header.startswith(('torch/', 'ATen/', 'c10/')):
            add('torch-core', match.start(), f'framework dependency in core: {header}')
        if header.startswith('cutlass/') and 'simt' in header.lower():
            add('simt-mma', match.start(), f'SIMT dependency in TensorOp family: {header}')
    for rule, pattern in PATTERNS.items():
        for match in pattern.finditer(code):
            add(rule, match.start(), f'local source uses {match.group().strip()}')
    return sorted(found.values(), key=lambda item: (item.line, item.rule))


def _excluded(parts: tuple[str, ...]) -> bool:
    # Root /build* follows the repository ignore rule. Do not exclude legitimate
    # local source names such as window_attention/builder.h or builders/.
    return any(p.lower() in EXCLUDED_DIRS for p in parts) or bool(parts and parts[0].lower().startswith('build'))


def _local_path(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError('paths must be nonempty repository-relative paths with / separators')
    relative = PurePosixPath(value)
    if relative.is_absolute() or '..' in relative.parts or ':' in value or any(c in value for c in '*?[]'):
        raise ValueError(f'not an exact repository-relative path: {value}')
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'path escapes repository: {value}')
    if _excluded(relative.parts):
        raise ValueError(f'path refers to excluded artifacts or third-party content: {value}')
    return path


def check_policy(repo_root: Path, policy_path: Path) -> Report:
    root, policy_path = repo_root.resolve(), policy_path.resolve()
    label = policy_path.as_posix()
    try:
        policy = json.loads(policy_path.read_text(encoding='utf-8-sig'))
        if not isinstance(policy, dict) or set(policy) != {'version', 'profile', 'source_roots', 'rules', 'exceptions'}:
            raise ValueError('expected version, profile, source_roots, rules and exceptions')
        if type(policy['version']) is not int or policy['version'] != 1 or not isinstance(policy['profile'], str) or not policy['profile'].strip():
            raise ValueError('expected version 1 and a nonempty profile')
        values = policy['rules']
        if not isinstance(values, list) or not values or any(not isinstance(r, str) for r in values):
            raise ValueError('rules must be a nonempty list of rule identifiers')
        rules = set(values)
        if len(rules) != len(values) or not rules <= RULES:
            raise ValueError(f'duplicate or unknown rule; available rules: {sorted(RULES)}')
        scopes = policy['source_roots']
        if not isinstance(scopes, list) or not scopes:
            raise ValueError('source_roots must be a nonempty list')
        files: set[Path] = set()
        for scope in scopes:
            directory = _local_path(root, scope)
            if not directory.is_dir():
                raise ValueError(f'missing source root: {scope}')
            selected = set()
            for path in directory.rglob('*'):
                relative = path.relative_to(root)
                if _excluded(relative.parts):
                    continue
                if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES:
                    if not path.resolve().is_relative_to(directory):
                        raise ValueError(f'source link escapes scope: {path}')
                    selected.add(path)
            if not selected:
                raise ValueError(f'empty source root: {scope}')
            files.update(selected)
        exceptions = policy['exceptions']
        if not isinstance(exceptions, list):
            raise ValueError('exceptions must be a list')
        registered: dict[tuple[str, str], str] = {}
        source_names = {p.relative_to(root).as_posix() for p in files}
        for entry in exceptions:
            if not isinstance(entry, dict) or set(entry) != {'path', 'rule', 'reason', 'decision'}:
                raise ValueError('exception requires exact path, rule, reason and decision document')
            if any(not isinstance(v, str) or not v.strip() for v in entry.values()):
                raise ValueError('exception fields must be nonempty strings')
            rule, path = entry['rule'], entry['path']
            if rule not in EXCEPTION_RULES or rule not in rules:
                raise ValueError(f'rule cannot be excepted: {rule}')
            _local_path(root, path)
            if path not in source_names:
                raise ValueError(f'exception does not name a scanned source file: {path}')
            decision = _local_path(root, entry['decision'])
            if decision.suffix != '.md' or not decision.is_file() or not decision.read_text(encoding='utf-8-sig').strip():
                raise ValueError(f'missing/non-markdown/empty decision document: {entry["decision"]}')
            key = (path, rule)
            if key in registered:
                raise ValueError(f'duplicate exception: {key}')
            registered[key] = entry['decision']
        diagnostics, allowed, used = [], [], set()
        for file in sorted(files):
            name = file.relative_to(root).as_posix()
            for diagnostic in scan_source(name, file.read_text(encoding='utf-8-sig'), rules):
                key = (name, diagnostic.rule)
                if key in registered:
                    used.add(key)
                    allowed.append(diagnostic)
                else:
                    diagnostics.append(diagnostic)
        for path, rule in registered.keys() - used:
            diagnostics.append(Diagnostic(label, 1, 'configuration', f'unused exception: {path} [{rule}]'))
        return Report(len(files), diagnostics, allowed)
    except (OSError, ValueError, TypeError) as error:
        return Report(0, [Diagnostic(label, 1, 'configuration', str(error))], [])


def enforce_policy(repo_root: Path, policy_path: Path) -> bool:
    report = check_policy(repo_root, policy_path)
    for diagnostic in report.diagnostics:
        print(diagnostic, file=sys.stderr, flush=True)
    for diagnostic in report.exceptions:
        print(f'SOURCE POLICY EXCEPTION (requires candidate review): {diagnostic}', flush=True)
    if report.diagnostics:
        print('SOURCE POLICY FAILED', file=sys.stderr, flush=True)
        return False
    print(f'SOURCE POLICY PASSED: {report.files} local source files; structure review still required', flush=True)
    return True
