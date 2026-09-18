"""Regex search of curated sources, optionally restricted to an architecture."""
import argparse
import re
from catalog import load_pages, normalize_arch, select


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pattern')
    parser.add_argument('--architecture')
    args = parser.parse_args()
    try:
        pattern = re.compile(args.pattern, re.I)
        arch = normalize_arch(args.architecture) if args.architecture else None
    except (ValueError, re.error) as error:
        parser.error(str(error))
    for p in select(load_pages(), architecture=arch):
        for line in p['body'].splitlines():
            if pattern.search(line):
                print(f"{p['path']}: {line}")


if __name__ == '__main__':
    main()
