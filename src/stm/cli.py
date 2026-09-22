"""``validate-stm``: validate STM manifests from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from stm.schema import Issue, is_valid, validate_catalogue, validate_title


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="validate-stm",
        description="Validate Signed Track Manifest files against the STM schema (0.1 and 0.2).",
    )
    p.add_argument("files", nargs="+", type=Path, help="manifest JSON files")
    p.add_argument(
        "--catalogue", action="store_true", help="validate as catalogue indexes, not titles"
    )
    p.add_argument(
        "--warnings-as-errors", action="store_true", help="exit non-zero on warnings too"
    )
    p.add_argument("-q", "--quiet", action="store_true", help="print only failures")
    return p


def _validate_file(path: Path, catalogue: bool) -> list[Issue]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [Issue("$", f"file not found: {path}")]
    except json.JSONDecodeError as exc:
        return [Issue("$", f"not valid JSON: {exc}")]
    return validate_catalogue(data) if catalogue else validate_title(data)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    failed = 0
    for path in args.files:
        issues = _validate_file(path, args.catalogue)
        ok = is_valid(issues, warnings_as_errors=args.warnings_as_errors)
        if not ok:
            failed += 1
        if ok and args.quiet:
            continue
        status = "ok" if ok else "FAIL"
        print(f"{path}: {status}")
        for issue in issues:
            print(f"  {issue}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
