"""``molexp.git.objects`` — real git OID framing, checked against the binary.

Spec: workspace-git-projection-02-objects. Every OID the framing computes
(blob / tree / commit) must be byte-identical to what the real ``git`` binary
produces in the same object database, the written objects must read back
through ``git cat-file``, and a constructed ref must walk its commit chain
under ``git log``. Commit OIDs must be deterministic and honour the
caller-supplied author/committer date (never ``now()``).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from molexp.git import (
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

SIG = Signature(name="Mol Exp", email="mol@exp.test", date="1700000000 +0000")


def _git(db_path: Path, *args: str, stdin: bytes | None = None) -> bytes:
    """Run the real ``git`` binary against ``db_path``; return raw stdout."""
    return subprocess.run(
        ["git", "-C", str(db_path), *args],
        input=stdin,
        capture_output=True,
        check=True,
    ).stdout


def _git_commit_tree(db_path: Path, tree_hex: str, message: str, sig: Signature) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": sig.name,
        "GIT_AUTHOR_EMAIL": sig.email,
        "GIT_AUTHOR_DATE": sig.date,
        "GIT_COMMITTER_NAME": sig.name,
        "GIT_COMMITTER_EMAIL": sig.email,
        "GIT_COMMITTER_DATE": sig.date,
    }
    return (
        subprocess.run(
            ["git", "-C", str(db_path), "commit-tree", tree_hex, "-m", message],
            capture_output=True,
            check=True,
            env=env,
        )
        .stdout.decode()
        .strip()
    )


# ── blob ─────────────────────────────────────────────────────────────────────


class TestBlob:
    async def test_blob_oid_matches_git_and_roundtrips(self, tmp_path):
        db = await ensure_object_db(tmp_path / "odb")
        data = b"hello molexp\n"
        oid = await write_blob(db, data)
        expected = _git(db.path, "hash-object", "-w", "--stdin", stdin=data).decode().strip()
        assert oid.hex == expected
        assert await read_object(db, oid) == data
        assert await object_type(db, oid) == "blob"

    async def test_binary_blob_roundtrips_lossless(self, tmp_path):
        db = await ensure_object_db(tmp_path / "odb")
        data = bytes(range(256))  # non-UTF-8 payload
        oid = await write_blob(db, data)
        expected = _git(db.path, "hash-object", "-w", "--stdin", stdin=data).decode().strip()
        assert oid.hex == expected
        assert await read_object(db, oid) == data


# ── tree ─────────────────────────────────────────────────────────────────────


class TestTree:
    async def test_tree_oid_matches_git_with_canonical_sort(self, tmp_path):
        db = await ensure_object_db(tmp_path / "odb")
        a = await write_blob(db, b"A")
        b = await write_blob(db, b"B")
        # Entries supplied OUT of order — build_tree must sort canonically.
        entries = [TreeEntry("100644", "beta.txt", b), TreeEntry("100644", "alpha.txt", a)]
        oid = await build_tree(db, entries)
        lines = "".join(
            f"100644 blob {e.oid.hex}\t{e.name}\n" for e in sorted(entries, key=lambda e: e.name)
        ).encode()
        expected = _git(db.path, "mktree", stdin=lines).decode().strip()
        assert oid.hex == expected
        assert await object_type(db, oid) == "tree"

    async def test_empty_tree_matches_git(self, tmp_path):
        db = await ensure_object_db(tmp_path / "odb")
        oid = await build_tree(db, [])
        expected = _git(db.path, "mktree", stdin=b"").decode().strip()
        assert oid.hex == expected
        # The well-known empty-tree OID.
        assert oid.hex == "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


# ── commit ───────────────────────────────────────────────────────────────────


class TestCommit:
    async def test_commit_oid_matches_git_and_is_deterministic(self, tmp_path):
        db = await ensure_object_db(tmp_path / "odb")
        blob = await write_blob(db, b"x")
        tree = await build_tree(db, [TreeEntry("100644", "f", blob)])
        oid1 = await build_commit(
            db, tree=tree, parents=[], message="root", author=SIG, committer=SIG
        )
        oid2 = await build_commit(
            db, tree=tree, parents=[], message="root", author=SIG, committer=SIG
        )
        assert oid1.hex == oid2.hex  # deterministic across calls
        assert oid1.hex == _git_commit_tree(db.path, tree.hex, "root", SIG)
        assert await object_type(db, oid1) == "commit"

    async def test_changing_only_the_date_changes_the_oid(self, tmp_path):
        db = await ensure_object_db(tmp_path / "odb")
        tree = await build_tree(db, [])
        early = Signature(name="N", email="e@x", date="1700000000 +0000")
        late = Signature(name="N", email="e@x", date="1800000000 +0000")
        c1 = await build_commit(
            db, tree=tree, parents=[], message="m", author=early, committer=early
        )
        c2 = await build_commit(db, tree=tree, parents=[], message="m", author=late, committer=late)
        assert c1.hex != c2.hex  # the date is hashed, not now()


# ── refs + chain walk ────────────────────────────────────────────────────────


class TestRefsAndWalk:
    async def test_multi_parent_commit_and_ref_walk(self, tmp_path):
        db = await ensure_object_db(tmp_path / "odb")
        tree = await build_tree(db, [])
        base = await build_commit(
            db, tree=tree, parents=[], message="base", author=SIG, committer=SIG
        )
        feat = await build_commit(
            db, tree=tree, parents=[base], message="feat", author=SIG, committer=SIG
        )
        other = await build_commit(
            db, tree=tree, parents=[base], message="other", author=SIG, committer=SIG
        )
        merge = await build_commit(
            db, tree=tree, parents=[feat, other], message="merge", author=SIG, committer=SIG
        )
        await set_ref(db, "refs/molexp/test", merge)
        got = await read_ref(db, "refs/molexp/test")
        assert got is not None
        assert got.hex == merge.hex
        # git log walks the whole reachable chain from the constructed ref.
        subjects = set(_git(db.path, "log", "--format=%s", "refs/molexp/test").decode().split())
        assert {"base", "feat", "other", "merge"} <= subjects

    async def test_read_ref_missing_returns_none(self, tmp_path):
        db = await ensure_object_db(tmp_path / "odb")
        assert await read_ref(db, "refs/molexp/does-not-exist") is None
