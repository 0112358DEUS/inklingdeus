#!/usr/bin/env python3
"""Verify the pinned E7 FA4 vendor tree byte-for-byte."""

from __future__ import annotations

import hashlib
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "third_party" / "inkling_sm120_fa4"
    manifest = root / "SHA256SUMS"
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        if name.startswith("/") or ".." in Path(name).parts:
            raise SystemExit(f"unsafe manifest path: {name}")
        expected[name] = digest
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest
    }
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise SystemExit(f"FA4 vendor file-set mismatch missing={missing} extra={extra}")
    for name, wanted in expected.items():
        got = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if got != wanted:
            raise SystemExit(f"FA4 vendor hash mismatch: {name} got={got} expected={wanted}")
    upstream = (root / "UPSTREAM_COMMIT").read_text(encoding="utf-8")
    if "60117041e10fcc6f19882afd274318c755a5ef6e" not in upstream:
        raise SystemExit("FA4 upstream commit record is missing")
    print(f"FA4 vendor PASS files={len(expected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
