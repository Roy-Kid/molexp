"""Content-agnostic git object framing — real OID computation.

The bottom primitive the workspace→git projection is built on. It computes
**real git object IDs** (blob / tree / commit) by driving the ``git`` binary
through the single :mod:`molexp.git._subprocess` chokepoint, so every object
produced is byte-identical to — and readable by — stock ``git``.

This module is deliberately **content-agnostic**: it knows nothing about
``molexp.workspace`` entities. The mapping from ``Folder`` / ``Asset`` /
``RunMetadata`` onto these primitives lives one layer up
(``workspace/git_projection.py``, spec 03).

Why a real OID and not :func:`molexp.ids.compute_content_hash`? They are
different functions: ``compute_content_hash`` is a bare ``sha256`` of bytes
(files) or a flat ``relpath\\0bytes`` digest (dirs); a git blob OID is
``sha1("blob <len>\\0" + bytes)`` and a tree OID is the recursive
``mode name\\0oid`` digest. The projection therefore computes *real* git
OIDs — a structural projection, not a hash relabelling. ``molexp.ids`` is
untouched by this layer.

Determinism is load-bearing for the projection's deterministic rebuild
(spec 03): commit author/committer name, email **and date** are all supplied
by the caller — never ``now()`` — so identical inputs yield a byte-identical
commit OID across processes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from molexp.git._subprocess import run_git, run_git_bytes

__all__ = [
    "ObjectDb",
    "Oid",
    "Signature",
    "TreeEntry",
    "build_commit",
    "build_tree",
    "ensure_object_db",
    "object_type",
    "read_object",
    "read_ref",
    "set_ref",
    "write_blob",
]

# git tree mode for a subtree entry (canonical form has no leading zero).
_TREE_MODES = frozenset({"40000", "040000"})
# git tree mode for a gitlink / submodule entry.
_COMMIT_MODE = "160000"


@dataclass(frozen=True)
class Oid:
    """A git object id (hex sha)."""

    hex: str

    def __str__(self) -> str:
        return self.hex


@dataclass(frozen=True)
class TreeEntry:
    """One entry of a git tree: ``mode`` + ``name`` + target ``oid``.

    ``mode`` is the git file mode string (``"100644"`` regular file,
    ``"100755"`` executable, ``"120000"`` symlink, ``"40000"`` subtree,
    ``"160000"`` gitlink).
    """

    mode: str
    name: str
    oid: Oid


@dataclass(frozen=True)
class ObjectDb:
    """Handle to a molexp-managed git object database (a bare repo)."""

    path: Path


@dataclass(frozen=True)
class Signature:
    """A git author/committer identity with a **caller-supplied** date.

    ``date`` is any value ``git`` accepts in ``GIT_AUTHOR_DATE`` /
    ``GIT_COMMITTER_DATE`` (e.g. ``"1700000000 +0000"`` or an RFC-2822 /
    ISO-8601 string). Supplying it explicitly — never ``now()`` — is what
    makes commit OIDs deterministic.
    """

    name: str
    email: str
    date: str


def _git_type_for_mode(mode: str) -> str:
    if mode in _TREE_MODES:
        return "tree"
    if mode == _COMMIT_MODE:
        return "commit"
    return "blob"


def _tree_sort_key(entry: TreeEntry) -> bytes:
    """git tree ordering: byte-wise on the name, subtree names sort as ``name/``."""
    name = entry.name
    if _git_type_for_mode(entry.mode) == "tree":
        name += "/"
    return name.encode("utf-8")


async def ensure_object_db(path: Path | str) -> ObjectDb:
    """Initialise (idempotently) a bare git object database at ``path``.

    A bare repo is a pure object store with no working tree — exactly what the
    projection needs. ``git init --bare`` is safe to re-run on an existing repo.
    """
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    await run_git(["init", "--bare", "--quiet", str(target)])
    return ObjectDb(path=target)


async def write_blob(db: ObjectDb, data: bytes) -> Oid:
    """Write ``data`` as a git blob and return its real OID (``hash-object -w``)."""
    res = await run_git_bytes(["hash-object", "-w", "--stdin"], cwd=db.path, stdin_data=data)
    return Oid(res.stdout.decode("ascii").strip())


async def build_tree(db: ObjectDb, entries: list[TreeEntry]) -> Oid:
    """Build a git tree from ``entries`` (canonically sorted) and return its OID."""
    lines = "".join(
        f"{e.mode} {_git_type_for_mode(e.mode)} {e.oid.hex}\t{e.name}\n"
        for e in sorted(entries, key=_tree_sort_key)
    )
    res = await run_git_bytes(["mktree"], cwd=db.path, stdin_data=lines.encode("utf-8"))
    return Oid(res.stdout.decode("ascii").strip())


async def build_commit(
    db: ObjectDb,
    *,
    tree: Oid,
    parents: list[Oid],
    message: str,
    author: Signature,
    committer: Signature,
) -> Oid:
    """Build a git commit over ``tree`` and return its real OID (``commit-tree``).

    Author/committer name, email and date are passed through to git via the
    ``GIT_*`` env vars — the date is the caller's, never ``now()`` — so the
    commit OID is deterministic for identical inputs.
    """
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_NAME": author.name,
            "GIT_AUTHOR_EMAIL": author.email,
            "GIT_AUTHOR_DATE": author.date,
            "GIT_COMMITTER_NAME": committer.name,
            "GIT_COMMITTER_EMAIL": committer.email,
            "GIT_COMMITTER_DATE": committer.date,
        }
    )
    args = ["commit-tree", tree.hex]
    for parent in parents:
        args += ["-p", parent.hex]
    args += ["-m", message]
    res = await run_git(args, cwd=db.path, env=env)
    return Oid(res.stdout.strip())


async def set_ref(db: ObjectDb, ref: str, oid: Oid) -> None:
    """Point ``ref`` (e.g. ``refs/molexp/...``) at ``oid`` (``update-ref``)."""
    await run_git(["update-ref", ref, oid.hex], cwd=db.path)


async def read_ref(db: ObjectDb, ref: str) -> Oid | None:
    """Resolve ``ref`` to its OID, or ``None`` when the ref does not exist."""
    res = await run_git(["rev-parse", "--verify", "--quiet", ref], cwd=db.path, check=False)
    resolved = res.stdout.strip()
    return Oid(resolved) if resolved else None


async def read_object(db: ObjectDb, oid: Oid) -> bytes:
    """Return the raw content of ``oid`` (``cat-file -p``), undecoded."""
    res = await run_git_bytes(["cat-file", "-p", oid.hex], cwd=db.path)
    return res.stdout


async def object_type(db: ObjectDb, oid: Oid) -> str:
    """Return the git object type of ``oid``: ``blob`` / ``tree`` / ``commit``."""
    res = await run_git(["cat-file", "-t", oid.hex], cwd=db.path)
    return res.stdout.strip()
