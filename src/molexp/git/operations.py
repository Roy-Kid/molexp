"""Generic git operations: clone, fetch, push.

Worktree management lives in :mod:`molexp.git.worktree`; everything else
goes here.
"""

from __future__ import annotations

from pathlib import Path

from molexp.git._subprocess import GitCommandError, run_git


async def ensure_clone(repo_url: str, target: Path) -> None:
    """Clone ``repo_url`` into ``target`` idempotently.

    If ``target`` already exists and is a git checkout (has a ``.git``
    file or directory), this is a no-op. If ``target`` exists but is
    *not* a git checkout, raises ``GitCommandError`` rather than
    blindly overwriting user content.

    Args:
        repo_url: Anything ``git clone`` accepts (https URL, git URL,
            local path).
        target: Destination directory. Created by ``git clone`` if it
            does not exist; left alone if it is already a git checkout.

    Raises:
        GitCommandError: On clone failure or when ``target`` exists with
            non-git content.
    """
    target = Path(target)
    if target.exists():
        if (target / ".git").exists():
            return  # already a checkout
        raise GitCommandError(
            ["clone", repo_url, str(target)],
            target.parent,
            None,
            f"target {target} already exists and is not a git checkout",
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    await run_git(["clone", repo_url, str(target)])


async def fetch(checkout: Path, *, depth: int | None = None) -> None:
    """Run ``git fetch`` inside ``checkout`` (optionally shallow).

    Args:
        checkout: Path to a git checkout.
        depth: If set, passes ``--depth=<n>`` for a shallow fetch.
    """
    args = ["fetch"]
    if depth is not None:
        args.append(f"--depth={depth}")
    await run_git(args, cwd=checkout)


async def push(checkout: Path, branch: str, *, remote: str = "origin") -> None:
    """Run ``git push <remote> <branch>`` inside ``checkout``.

    Args:
        checkout: Path to a git checkout (or worktree).
        branch: Branch ref to push.
        remote: Remote name; defaults to ``origin``.
    """
    await run_git(["push", remote, branch], cwd=checkout)
