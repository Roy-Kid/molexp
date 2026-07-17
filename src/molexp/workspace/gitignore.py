"""``.gitignore``-aware path filter for workspace tree walks.

Scientific workspaces are often git-managed: dependency trees, build
outputs, and editor caches already live in ``.gitignore``. Bundle
enumeration (``GET /api/knowledge``) and the UI file browser
(``GET /api/workspace/files``) share this matcher so they skip the same
paths git would — without re-implementing a second hard-coded denylist.

Semantics (intentionally a practical subset of git):

* Always apply a small **safety floor** of patterns (``.git/``,
  ``node_modules/``, venvs, caches…) even when no ``.gitignore`` exists —
  so an un-inited tree cannot ENAMETOOLONG through cyclic package
  symlinks.
* Load the root ``.gitignore`` and any nested ``.gitignore`` files along
  a path (git cascade); within each file last-match wins, including
  negations (``!keep.me``).
* Directory-only patterns (trailing ``/``) are honoured by matching both
  ``name`` and ``name/`` when the caller marks ``is_dir=True``.
* Re-including a path *inside* an already-ignored parent (git's
  ``!node_modules/keep/`` after ignoring ``node_modules/``) is **not**
  supported: once a directory is ignored we do not descend. That matches
  what the UI and knowledge walks need and keeps walks O(visible tree).

The matcher is filesystem-agnostic: it reads via the workspace
:class:`~molexp.workspace.fs.FileSystem` Protocol so remote mirrors work.
"""

from __future__ import annotations

import os
from pathlib import PurePosixPath

from pathspec import GitIgnoreSpec

from .fs import FileSystem, PathArg
from .fs_local import LocalFileSystem

__all__ = [
    "DEFAULT_IGNORE_LINES",
    "GitIgnoreMatcher",
    "load_gitignore_matcher",
]

# Safety floor — always applied, even with no on-disk .gitignore. Keep in
# sync with the spirit of agent tree skips (ops.structure / interactive
# tools); this is the single server/workspace source of truth.
DEFAULT_IGNORE_LINES: tuple[str, ...] = (
    ".git/",
    ".hg/",
    ".svn/",
    "__pycache__/",
    "node_modules/",
    ".venv/",
    "venv/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".pytest_cache/",
    ".tox/",
    ".eggs/",
    "agent/.scratch/",
    ".benchmarks/",
    "*.egg-info/",
)


def _norm_rel(path: str) -> str:
    """POSIX-relative path with no leading/trailing slash ('' = root)."""
    return path.replace("\\", "/").strip("/")


class GitIgnoreMatcher:
    """Cascade of default + root + nested ``.gitignore`` specs for one root."""

    def __init__(self, root: PathArg, *, fs: FileSystem | None = None) -> None:
        self._root = os.fspath(root)
        self._fs: FileSystem = fs if fs is not None else LocalFileSystem()
        # (base_rel, spec) — base_rel is '' for root-level rules; nested
        # patterns apply relative to their owning directory.
        self._layers: list[tuple[str, GitIgnoreSpec]] = [
            ("", GitIgnoreSpec.from_lines(DEFAULT_IGNORE_LINES)),
        ]
        self._loaded_bases: set[str] = set()
        self._try_load("")

    @property
    def root(self) -> str:
        return self._root

    def _gitignore_abs(self, base_rel: str) -> str:
        if base_rel:
            return self._fs.join(self._root, *base_rel.split("/"), ".gitignore")
        return self._fs.join(self._root, ".gitignore")

    def _try_load(self, base_rel: str) -> None:
        """Load ``base_rel/.gitignore`` once (no-op if missing / unreadable)."""
        base = _norm_rel(base_rel)
        if base in self._loaded_bases:
            return
        self._loaded_bases.add(base)
        path = self._gitignore_abs(base)
        try:
            if not self._fs.is_file(path):
                return
            text = self._fs.read_text(path)
        except OSError:
            return
        lines = text.splitlines()
        if lines:
            self._layers.append((base, GitIgnoreSpec.from_lines(lines)))

    def ensure_ancestors(self, rel_path: str) -> None:
        """Load every nested ``.gitignore`` from root down to *rel_path*'s dir."""
        rel = _norm_rel(rel_path)
        self._try_load("")
        if not rel:
            return
        parts = rel.split("/")
        # Load gitignores for each ancestor directory (not the leaf file itself
        # when rel is a file — caller passes the parent, or a dir path).
        acc: list[str] = []
        for part in parts:
            acc.append(part)
            self._try_load("/".join(acc))

    def is_ignored(self, rel_path: str, *, is_dir: bool = False) -> bool:
        """Return ``True`` if *rel_path* (workspace-relative) should be skipped.

        Args:
            rel_path: Path relative to the matcher root (POSIX or OS-native).
            is_dir: Whether *rel_path* names a directory — enables directory-
                only patterns that end with ``/``.
        """
        rel = _norm_rel(rel_path)
        if not rel:
            return False

        parent = str(PurePosixPath(rel).parent)
        self.ensure_ancestors("" if parent == "." else parent)
        # Also load the dir's own .gitignore when checking a directory so
        # children evaluated next see nested rules; the dir itself is still
        # decided by ancestor rules only (see loop below).
        if is_dir:
            self._try_load(rel)

        ignored = False
        for base, spec in self._layers:
            if base:
                if rel == base:
                    # A directory is not matched against its own .gitignore.
                    continue
                if not rel.startswith(base + "/"):
                    continue
                local = rel[len(base) + 1 :]
            else:
                local = rel

            decision = self._check(spec, local, is_dir=is_dir)
            if decision is True:
                ignored = True
            elif decision is False:
                ignored = False
        return ignored

    @staticmethod
    def _check(spec: GitIgnoreSpec, local: str, *, is_dir: bool) -> bool | None:
        """Return True/False if a pattern matched (ignored / un-ignored), else None."""
        # Directory-only patterns (``foo/``) need a trailing slash; file
        # patterns match the bare path. Prefer an explicit hit.
        if is_dir:
            dir_result = spec.check_file(local + "/")
            if dir_result.include is not None:
                return bool(dir_result.include)
        result = spec.check_file(local)
        if result.include is not None:
            return bool(result.include)
        return None


def load_gitignore_matcher(root: PathArg, *, fs: FileSystem | None = None) -> GitIgnoreMatcher:
    """Build a :class:`GitIgnoreMatcher` for *root* (lazy nested loads)."""
    return GitIgnoreMatcher(root, fs=fs)
