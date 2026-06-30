"""``GitWorktreeManager`` — per-issue git worktree lifecycle.

Each "issue" in symphony (or any consumer) gets its own working
directory on its own branch, but they all share a single ``.git/``
object database via ``git worktree``. Cheap on disk vs. one full clone
per issue.
"""

from __future__ import annotations

from builtins import list as _list
from pathlib import Path

from molexp.git._subprocess import run_git


class GitWorktreeManager:
    """Manage worktrees attached to a single underlying git checkout.

    The manager is bound to ``root`` — the "main" checkout where ``.git``
    lives. Worktrees are external directories that share that ``.git``
    object DB.

    Args:
        root: Path to a git checkout. Worktree commands run inside it.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    async def add(self, branch: str, worktree_path: Path) -> None:
        """Create a new worktree at ``worktree_path`` on ``branch``.

        Equivalent to::

            git -C <root> worktree add <worktree_path> -b <branch>

        ``branch`` is created if it does not exist; if the branch
        already exists, the call fails (use git directly to attach to
        an existing branch).
        """
        await run_git(
            ["worktree", "add", str(worktree_path), "-b", branch],
            cwd=self.root,
        )

    async def add_detached(self, commit: str, worktree_path: Path) -> None:
        """Materialize ``commit`` into ``worktree_path`` in detached HEAD.

        Equivalent to::

            git -C <root> worktree add --detach <worktree_path> <commit>

        ``commit`` may be a commit OID or any ref (e.g. ``refs/molexp/runs/<id>``).
        Unlike :meth:`add` it creates no branch — used to materialize a
        historical state into a scratch dir for inspection / re-run, never into
        the live workspace.
        """
        await run_git(
            ["worktree", "add", "--detach", str(worktree_path), commit],
            cwd=self.root,
        )

    async def remove(self, worktree_path: Path, *, force: bool = False) -> None:
        """Tear down ``worktree_path`` and detach it from the main repo.

        Args:
            worktree_path: Worktree directory to remove.
            force: Forward ``--force`` so removal succeeds even with
                local edits / untracked files inside the worktree.
        """
        args = ["worktree", "remove", str(worktree_path)]
        if force:
            args.append("--force")
        await run_git(args, cwd=self.root)

    async def list(self) -> _list[Path]:
        """Return the paths of all worktrees attached to ``root``.

        The main checkout itself is included as the first entry.
        """
        result = await run_git(["worktree", "list", "--porcelain"], cwd=self.root)
        paths: list[Path] = []
        for line in result.stdout.splitlines():
            if line.startswith("worktree "):
                paths.append(Path(line[len("worktree ") :]))
        return paths

    async def prune(self) -> None:
        """Drop git's bookkeeping for worktree dirs that have been deleted
        from disk (``rm -rf <worktree>``). Idempotent."""
        await run_git(["worktree", "prune"], cwd=self.root)
