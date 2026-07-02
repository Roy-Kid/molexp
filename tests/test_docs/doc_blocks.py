"""Collector for executable ``python`` code blocks in ``docs/**/*.md``.

Every fenced ```` ```python ```` block in the documentation is extracted as
one :class:`DocBlock` so ``test_doc_code_blocks.py`` can execute it as a
pytest case. The contract for doc authors:

- Blocks on one page share a namespace and a working directory, and run in
  page order — tutorial pages may build state progressively.
- A block whose **first non-empty line** is ``# docs: skip — <reason>`` is
  collected but never executed (pseudo-code, LLM-key/network-dependent
  snippets). The reason is mandatory.
- A block whose first non-empty line is ``# docs: xfail — <reason>`` is
  executed but expected to fail until the referenced workflow lands
  (``strict=False``: it turns green silently at integration time).
- Top-level ``await`` is allowed (blocks are compiled with
  ``PyCF_ALLOW_TOP_LEVEL_AWAIT`` and driven by ``asyncio.run``), so docs can
  show idiomatic async driver code without ``asyncio.run`` boilerplate.

See ``tests/test_docs/README.md`` for the full convention.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs"

# Languages treated as executable Python. ``pycon`` (>>> transcripts) is
# deliberately NOT collected — use plain ``python`` blocks for tested code.
_PYTHON_LANGS = {"python", "py", "python3"}

# Separator accepts hyphen, en dash, em dash, or colon between marker and reason.
_MARKER_RE = re.compile(
    r"^#\s*docs:\s*(?P<kind>skip|xfail)\s*(?:[-–—:]\s*(?P<reason>.+))?\s*$"  # noqa: RUF001
)


@dataclass(frozen=True)
class DocBlock:
    """One fenced ```python block from a docs page."""

    page: str  # path relative to the repo root, POSIX style
    index: int  # 0-based position among the page's python blocks
    lineno: int  # 1-based line number of the opening fence
    source: str
    skip_reason: str | None = None
    xfail_reason: str | None = None

    @property
    def test_id(self) -> str:
        return f"{self.page}:b{self.index:02d}"


def _parse_marker(source: str) -> tuple[str | None, str | None]:
    """Return ``(skip_reason, xfail_reason)`` from the block's first line."""
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _MARKER_RE.match(stripped)
        if match is None:
            return None, None
        reason = match.group("reason") or "unspecified (add one!)"
        if match.group("kind") == "skip":
            return reason, None
        return None, reason
    return None, None


def _blocks_in_page(md_path: Path) -> list[DocBlock]:
    rel = md_path.relative_to(REPO_ROOT).as_posix()
    blocks: list[DocBlock] = []
    lines = md_path.read_text(encoding="utf-8").splitlines()
    in_block = False
    fence_indent = ""
    fence_lineno = 0
    is_python = False
    buf: list[str] = []
    for lineno, raw in enumerate(lines, start=1):
        stripped = raw.lstrip()
        if not in_block:
            if stripped.startswith("```"):
                in_block = True
                fence_indent = raw[: len(raw) - len(stripped)]
                fence_lineno = lineno
                info = stripped[3:].strip()
                lang = info.split()[0].lower() if info else ""
                is_python = lang in _PYTHON_LANGS
                buf = []
            continue
        if stripped.startswith("```") and stripped.rstrip("`") == "":
            in_block = False
            if is_python:
                source = textwrap.dedent("\n".join(buf))
                skip_reason, xfail_reason = _parse_marker(source)
                blocks.append(
                    DocBlock(
                        page=rel,
                        index=len(blocks),
                        lineno=fence_lineno,
                        source=source,
                        skip_reason=skip_reason,
                        xfail_reason=xfail_reason,
                    )
                )
            continue
        if is_python:
            # Strip the admonition indent so nested blocks compile.
            buf.append(raw.removeprefix(fence_indent) if fence_indent else raw)
    return blocks


def collect_doc_blocks() -> list[DocBlock]:
    """All python blocks from every docs page, page-ordered."""
    out: list[DocBlock] = []
    for md_path in sorted(DOCS_ROOT.rglob("*.md")):
        out.extend(_blocks_in_page(md_path))
    return out
