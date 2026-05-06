"""Tests for ``molexp.git.GitWorktreeManager``.

Each test sets up a real bare git repo in ``tmp_path`` (so worktree
operations exercise the actual ``git worktree`` plumbing) and exercises
the async manager API. ``git`` binary on PATH is required.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def _init_seed_repo(tmp_path: Path) -> Path:
    """Init a non-bare repo with one commit; return its working dir.

    The ``GitWorktreeManager`` operates on this checkout; its worktrees
    are added beside the checkout, sharing the underlying ``.git/``.
    """
    seed = tmp_path / "seed"
    seed.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(seed)], check=True)
    subprocess.run(["git", "-C", str(seed), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(seed), "config", "user.name", "Test"], check=True)
    (seed / "README.md").write_text("seed\n")
    subprocess.run(["git", "-C", str(seed), "add", "README.md"], check=True)
    subprocess.run(
        ["git", "-C", str(seed), "commit", "-q", "-m", "seed"],
        check=True,
        env={
            "GIT_AUTHOR_NAME": "T",
            "GIT_AUTHOR_EMAIL": "t@x",
            "GIT_COMMITTER_NAME": "T",
            "GIT_COMMITTER_EMAIL": "t@x",
            "PATH": "/usr/bin:/bin:/usr/local/bin",
        },
    )
    return seed


@pytest.mark.asyncio
async def test_add_creates_shared_worktree(tmp_path: Path):
    from molexp.git import GitWorktreeManager

    seed = _init_seed_repo(tmp_path)
    wt_path = tmp_path / "wt-issue-1"
    mgr = GitWorktreeManager(seed)

    await mgr.add("claude/issue-1", wt_path)

    # Worktree directory exists with a checkout.
    assert wt_path.is_dir()
    assert (wt_path / "README.md").is_file()
    # The .git in the worktree is a *file* (gitlink), not a dir — proof
    # the object DB is shared with the seed repo.
    git_pointer = wt_path / ".git"
    assert git_pointer.exists()
    assert git_pointer.is_file(), ".git in a worktree should be a gitlink file, not a dir"
    assert "gitdir:" in git_pointer.read_text()


@pytest.mark.asyncio
async def test_add_lists_then_remove_and_prune(tmp_path: Path):
    from molexp.git import GitWorktreeManager

    seed = _init_seed_repo(tmp_path)
    wt_path = tmp_path / "wt-issue-2"
    mgr = GitWorktreeManager(seed)

    await mgr.add("claude/issue-2", wt_path)

    listed = await mgr.list()
    listed_str = {str(p) for p in listed}
    assert str(wt_path) in listed_str

    await mgr.remove(wt_path)
    assert not wt_path.exists()

    # rm -rf'd a worktree dir (simulate user manual delete) → prune cleans
    other = tmp_path / "wt-stale"
    await mgr.add("claude/stale", other)
    import shutil

    shutil.rmtree(other)
    await mgr.prune()
    listed_after = await mgr.list()
    assert all("wt-stale" not in str(p) for p in listed_after)


@pytest.mark.asyncio
async def test_remove_and_prune(tmp_path: Path):
    """Combined: alias for the canonical close-out check (ac-002)."""
    from molexp.git import GitWorktreeManager

    seed = _init_seed_repo(tmp_path)
    mgr = GitWorktreeManager(seed)
    wt_path = tmp_path / "wt"
    await mgr.add("claude/x", wt_path)
    await mgr.remove(wt_path)
    await mgr.prune()
    assert not wt_path.exists()
    listed = await mgr.list()
    assert all("/wt" not in str(p) or "wt-" in str(p) for p in listed)
