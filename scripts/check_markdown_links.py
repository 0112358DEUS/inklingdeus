#!/usr/bin/env python3
"""Check that local file targets in Markdown links exist."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote


INLINE_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
REFERENCE_LINK = re.compile(r"^\s*\[[^\]]+\]:\s+(\S+)", re.MULTILINE)
SKIP_PREFIXES = ("#", "http://", "https://", "mailto:", "data:")


def normalize_destination(raw: str) -> str:
    destination = raw.strip()
    if destination.startswith("<") and ">" in destination:
        destination = destination[1 : destination.index(">")]
    else:
        destination = destination.split(maxsplit=1)[0]
    return unquote(destination.split("#", 1)[0].split("?", 1)[0])


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []
    checked = 0
    for markdown in sorted(root.rglob("*.md")):
        if ".git" in markdown.parts:
            continue
        text = markdown.read_text(encoding="utf-8")
        destinations = INLINE_LINK.findall(text) + REFERENCE_LINK.findall(text)
        for raw in destinations:
            if raw.startswith(SKIP_PREFIXES):
                continue
            destination = normalize_destination(raw)
            if not destination:
                continue
            checked += 1
            target = (markdown.parent / destination).resolve()
            if not target.exists():
                failures.append(
                    f"{markdown.relative_to(root)}: missing local link {raw!r}"
                )
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"markdown links PASS ({checked} local targets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
