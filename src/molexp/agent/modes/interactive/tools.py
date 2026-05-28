"""Read-only tool set for the emergent :class:`InteractiveMode` loop.

Three plain-Python callables — :func:`read_file`, :func:`list_directory`,
:func:`search_code` — handed verbatim to pydantic-ai's
``Agent(tools=[...])``. The model may *inspect* the workspace; it may
not change it. v1 ships **no** write / edit / shell tool: write-side
orchestration lives in :mod:`molexp.harness`, not in InteractiveMode.

Every tool is confined to one workspace root. A path that is absolute,
contains a ``..`` component, or otherwise resolves outside the root is
rejected with :exc:`ValueError` before any I/O happens.

This module imports nothing from ``pydantic_ai`` — the tools are bare
callables the SDK introspects by signature + docstring. :func:`readonly_tools`
closes over the workspace root so the model never sees it as an argument.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

__all__ = ["readonly_tools"]

# Directories never walked by ``search_code`` / hidden from ``list_directory``
# scans — version-control, build, cache, and dependency trees.
_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".agent-scratch",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "dist",
        "build",
    }
)

_MAX_FILE_BYTES = 512 * 1024
"""Largest file ``read_file`` / ``search_code`` will touch — keeps a
single tool call's output bounded."""

_MAX_SEARCH_HITS = 60
"""Cap on ``search_code`` match rows so one call cannot flood context."""


def _safe_path(root: Path, path: str) -> Path:
    """Resolve ``path`` against ``root``, rejecting any escape outside it.

    Args:
        root: The workspace root every tool is confined to.
        path: A caller-supplied, workspace-relative path.

    Returns:
        The resolved absolute path, guaranteed to be ``root`` itself or
        a descendant of it.

    Raises:
        ValueError: If ``path`` is absolute, contains a ``..`` segment,
            or resolves outside ``root``.
    """
    raw = Path(path)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"path {path!r} must be workspace-relative and may not contain '..'")
    root_resolved = root.resolve()
    resolved = (root_resolved / raw).resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ValueError(f"path {path!r} escapes the workspace root {root_resolved}")
    return resolved


def _read_text(file_path: Path) -> str:
    """Read a UTF-8 text file, rejecting oversized / binary content."""
    size = file_path.stat().st_size
    if size > _MAX_FILE_BYTES:
        raise ValueError(
            f"{file_path.name} is {size} bytes; refusing to read more than {_MAX_FILE_BYTES}"
        )
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{file_path.name} is not a UTF-8 text file") from exc


def readonly_tools(workspace_root: Path) -> tuple[Callable[..., str], ...]:
    """Build the read-only tool callables confined to ``workspace_root``.

    The returned callables close over the root, so pydantic-ai only ever
    introspects model-facing arguments (``path`` / ``pattern``) — never
    the root itself.

    Args:
        workspace_root: Directory every tool is confined to.

    Returns:
        ``(read_file, list_directory, search_code)`` — exactly three
        read-only tools, in a stable order.
    """
    root = Path(workspace_root)

    def read_file(path: str) -> str:
        """Read a UTF-8 text file from the workspace and return its contents.

        Args:
            path: Workspace-relative path to the file, e.g.
                ``"src/molexp/agent/runner.py"``.
        """
        target = _safe_path(root, path)
        if not target.is_file():
            raise FileNotFoundError(f"no such file in the workspace: {path!r}")
        return _read_text(target)

    def list_directory(path: str = ".") -> str:
        """List the entries of a directory inside the workspace.

        Args:
            path: Workspace-relative directory path; defaults to the
                workspace root.
        """
        target = _safe_path(root, path)
        if not target.is_dir():
            raise NotADirectoryError(f"no such directory in the workspace: {path!r}")
        rows: list[str] = []
        for entry in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name)):
            if entry.name in _SKIP_DIRS:
                continue
            marker = "/" if entry.is_dir() else ""
            rows.append(f"{entry.name}{marker}")
        return "\n".join(rows) if rows else "(empty directory)"

    def search_code(pattern: str) -> str:
        """Search workspace text files for a regular-expression pattern.

        Args:
            pattern: A Python regular expression to search for.

        Returns:
            Matching ``relative/path:lineno: line`` rows (capped), or a
            note when nothing matched.
        """
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"invalid search pattern {pattern!r}: {exc}") from exc
        root_resolved = root.resolve()
        hits: list[str] = []
        for file_path in sorted(root_resolved.rglob("*")):
            if len(hits) >= _MAX_SEARCH_HITS:
                hits.append(f"... (truncated at {_MAX_SEARCH_HITS} matches)")
                break
            if not file_path.is_file():
                continue
            if any(part in _SKIP_DIRS for part in file_path.relative_to(root_resolved).parts):
                continue
            try:
                if file_path.stat().st_size > _MAX_FILE_BYTES:
                    continue
                text = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            rel = file_path.relative_to(root_resolved).as_posix()
            for lineno, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    hits.append(f"{rel}:{lineno}: {line.strip()}")
                    if len(hits) >= _MAX_SEARCH_HITS:
                        break
        return "\n".join(hits) if hits else f"no matches for {pattern!r}"

    return (read_file, list_directory, search_code)
