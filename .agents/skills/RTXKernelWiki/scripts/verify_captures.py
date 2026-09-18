"""Offline verification of downloaded source bytes and capture provenance."""
import argparse
import hashlib
from pathlib import Path
import yaml
from catalog import ROOT
from github_capture import safe_path


def verify_bundle(bundle):
    errors = []
    manifest = yaml.safe_load((bundle / 'PROVENANCE.yaml').read_text(encoding='utf-8'))
    if not manifest['complete']:
        errors.append('capture marked incomplete: ' + str(manifest.get('errors', []) + manifest.get('truncations', []) + manifest.get('skipped', [])))
    for item in manifest['files']:
        try:
            data = safe_path(bundle, item['path']).read_bytes()
            if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
                errors.append(f'{item["path"]}: size/hash mismatch')
        except (OSError, ValueError) as error:
            errors.append(str(error))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, nargs='?', default=ROOT / 'artifacts/github')
    args = parser.parse_args()
    manifests = list(args.directory.rglob('PROVENANCE.yaml'))
    if not manifests:
        parser.error('No captures found')
    failures = []
    for path in manifests:
        try:
            failures.extend(f'{path.parent}: {error}' for error in verify_bundle(path.parent))
        except (OSError, KeyError, ValueError, yaml.YAMLError) as error:
            failures.append(f'{path}: {error}')
    if failures:
        print('\n'.join(failures))
        raise SystemExit(1)
    print(f'Verified {len(manifests)} captures: all recorded files, lengths and SHA256 hashes match.')


if __name__ == '__main__':
    main()
