"""Tests for ``molexp.git.operations.ensure_clone``.

``ensure_clone`` is the only operation exercised here; ``fetch`` and
``push`` interact with remotes and are out of scope for unit tests
(they get smoke coverage at integration time).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from molexp.git import ensure_clone


def _make_remote(tmp_path: Path) -> Path:
    """Create a bare repo with one commit to act as the upstream remote."""
    upstream = tmp_path / "upstream.git"
    subprocess.run(
        ["git", "init", "-q", "--bare", "-b", "main", str(upstream)],
        check=True,
    )
    seed = tmp_path / "seed"
    seed.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(seed)], check=True)
    subprocess.run(["git", "-C", str(seed), "config", "user.email", "t@x"], check=True)
    subprocess.run(["git", "-C", str(seed), "config", "user.name", "T"], check=True)
    (seed / "f.txt").write_text("hello")
    subprocess.run(["git", "-C", str(seed), "add", "f.txt"], check=True)
    subprocess.run(["git", "-C", str(seed), "commit", "-q", "-m", "init"], check=True)
    subprocess.run(["git", "-C", str(seed), "remote", "add", "origin", str(upstream)], check=True)
    subprocess.run(["git", "-C", str(seed), "push", "-q", "origin", "main"], check=True)
    return upstream


class TestEnsureClone:
    async def test_second_call_is_a_noop_on_existing_checkout(self, tmp_path: Path):
        remote = _make_remote(tmp_path)
        target = tmp_path / "checkout"

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

    async def test_rejects_non_git_dir_in_target(self, tmp_path: Path):
        """A pre-existing non-git ``target`` is refused, never blasted."""
        remote = _make_remote(tmp_path)
        target = tmp_path / "checkout"
        target.mkdir()
        (target / "stranger.txt").write_text("not from git")

        with pytest.raises(Exception):  # noqa: B017
            await ensure_clone(str(remote), target)
