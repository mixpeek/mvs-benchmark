#!/usr/bin/env python3
"""Validate benchmark result JSON files before publishing or comparing them."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.validation import UnsupportedScenario, ValidationError, validate_result_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MVS benchmark result JSON files")
    parser.add_argument("paths", nargs="+", help="Result JSON file or directory paths")
    return parser.parse_args()


def iter_json_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.json")))
        else:
            files.append(path)
    return files


def main() -> int:
    failures = 0
    for path in iter_json_files(parse_args().paths):
        try:
            validate_result_file(path)
        except UnsupportedScenario as exc:
            print(f"SKIP {path}: {exc}")
        except (OSError, ValidationError) as exc:
            failures += 1
            print(f"FAIL {path}: {exc}")
        else:
            print(f"OK   {path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
