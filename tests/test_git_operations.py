"""Tests for generic git operations in ``molexp.git.operations``.

``ensure_clone`` is the only operation we exercise here; ``fetch`` and
``push`` interact with remotes and are out of scope for unit tests
(would need a real remote or a heavier fixture). They get smoke
coverage at integration time.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def _make_remote(tmp_path: Path) -> Path:
    """Create a bare repo to act as the upstream remote."""
    upstream = tmp_path / "upstream.git"
    subprocess.run(
        ["git", "init", "-q", "--bare", "-b", "main", str(upstream)],
        check=True,
    )
    # Push one commit so a clone has something to checkout.
    seed = tmp_path / "seed"
    seed.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(seed)], check=True)
    subprocess.run(["git", "-C", str(seed), "config", "user.email", "t@x"], check=True)
    subprocess.run(["git", "-C", str(seed), "config", "user.name", "T"], check=True)
    (seed / "f.txt").write_text("hello")
    subprocess.run(["git", "-C", str(seed), "add", "f.txt"], check=True)
    subprocess.run(["git", "-C", str(seed), "commit", "-q", "-m", "init"], check=True)
    subprocess.run(
        ["git", "-C", str(seed), "remote", "add", "origin", str(upstream)], check=True
    )
    subprocess.run(
        ["git", "-C", str(seed), "push", "-q", "origin", "main"], check=True
    )
    return upstream


@pytest.mark.asyncio
async def test_ensure_clone_idempotent(tmp_path: Path):
    from molexp.git import ensure_clone

    remote = _make_remote(tmp_path)
    target = tmp_path / "checkout"

    # First call clones.
    await ensure_clone(str(remote), target)
    assert target.is_dir()
    assert (target / "f.txt").is_file()
    head_first = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    # Second call is a no-op (target already exists with .git).
    await ensure_clone(str(remote), target)
    head_second = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert head_first == head_second


@pytest.mark.asyncio
async def test_ensure_clone_rejects_non_git_dir_in_target(tmp_path: Path):
    """If `target` exists but is not a git checkout, refuse — don't blast it."""
    from molexp.git import ensure_clone

    remote = _make_remote(tmp_path)
    target = tmp_path / "checkout"
    target.mkdir()
    (target / "stranger.txt").write_text("not from git")

    with pytest.raises(Exception):
        await ensure_clone(str(remote), target)
