"""Generic git operations for molexp workspaces.

Core (not a plugin) because git is a hard infrastructure dependency for
molexp's experiment / workflow lifecycle: every project ultimately
needs to clone, branch, and isolate working directories.

The public surface is intentionally narrow:

- :func:`ensure_clone` — idempotent ``git clone`` into a target dir.
- :func:`fetch` — ``git fetch`` against a checkout.
- :func:`push` — ``git push`` from a checkout.
- :class:`GitWorktreeManager` — ``git worktree add/remove/list/prune``
  for the per-issue worktree pattern (one shared object DB, many
  lightweight working dirs on different branches).

Implementation strategy: thin async wrappers around the ``git`` binary
via ``asyncio.create_subprocess_exec``. No third-party git library is
introduced (gitpython adds a sync dep that itself wraps the same
binary; pygit2 needs libgit2 C bindings).
"""

from __future__ import annotations

from molexp.git.objects import (
    ObjectDb,
    Oid,
    Signature,
    TreeEntry,
    build_commit,
    build_tree,
    ensure_object_db,
    object_type,
    read_object,
    read_ref,
    set_ref,
    write_blob,
)
from molexp.git.operations import ensure_clone, fetch, push
from molexp.git.worktree import GitWorktreeManager

__all__ = [
    "GitWorktreeManager",
    "ObjectDb",
    "Oid",
    "Signature",
    "TreeEntry",
    "build_commit",
    "build_tree",
    "ensure_clone",
    "ensure_object_db",
    "fetch",
    "object_type",
    "push",
    "read_object",
    "read_ref",
    "set_ref",
    "write_blob",
]
