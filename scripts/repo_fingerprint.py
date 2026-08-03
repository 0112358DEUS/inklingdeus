#!/usr/bin/env python3
"""Deterministically fingerprint the runnable repository payload, including untracked files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Iterable


EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "artifacts",
    "node_modules",
}
EXCLUDED_FILES = {".DS_Store", ".git"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def included_paths(root: Path) -> Iterable[Path]:
    for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
        dirs[:] = sorted(name for name in dirs if name not in EXCLUDED_DIRS)
        current_path = Path(current)
        for name in sorted(files):
            path = current_path / name
            if name in EXCLUDED_FILES or path.suffix in EXCLUDED_SUFFIXES:
                continue
            yield path


def fingerprint(root: Path) -> dict[str, object]:
    root = root.resolve()
    digest = hashlib.sha256()
    entries = []
    for path in included_paths(root):
        relative = path.relative_to(root).as_posix()
        mode = stat.S_IMODE(path.lstat().st_mode)
        if path.is_symlink():
            kind = "symlink"
            content = os.readlink(path).encode()
        elif path.is_file():
            kind = "file"
            content = path.read_bytes()
        else:
            continue
        file_hash = hashlib.sha256(content).hexdigest()
        header = f"{kind}\0{mode:o}\0{relative}\0{len(content)}\0{file_hash}\n".encode()
        digest.update(header)
        entries.append(
            {
                "path": relative,
                "kind": kind,
                "mode": f"{mode:o}",
                "bytes": len(content),
                "sha256": file_hash,
            }
        )
    return {
        "schema_version": 1,
        "root": str(root),
        "files": len(entries),
        "sha256": digest.hexdigest(),
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--digest-only", action="store_true")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    payload = fingerprint(args.root)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.digest_only:
        print(payload["sha256"])
    else:
        print(json.dumps({key: payload[key] for key in ("schema_version", "files", "sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
