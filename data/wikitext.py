"""Dataset loading helpers for WikiText-2 line-level language modeling."""

from __future__ import annotations

from pathlib import Path


def read_wikitext_lines(path: str | Path, max_lines: int | None = None) -> list[str]:
    lines: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        text = raw.rstrip("\n")
        if not text.strip():
            continue
        lines.append(text + "\n")
        if max_lines is not None and len(lines) >= max_lines:
            break
    return lines
