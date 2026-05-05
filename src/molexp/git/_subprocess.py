"""Async ``git`` subprocess helper.

Single chokepoint for ``asyncio.create_subprocess_exec("git", ...)`` so
all git invocations in molexp share a uniform timeout, error shape, and
stdout/stderr capture policy.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TIMEOUT_S = 120.0


class GitCommandError(RuntimeError):
    """Raised when ``git`` exits non-zero or times out.

    Attributes:
        args: The ``git`` argv used (without the leading ``"git"``).
        cwd: Working directory the command ran in.
        returncode: Exit code, or ``None`` if the run timed out.
        stderr: Captured stderr (decoded, may be empty).
    """

    def __init__(
        self,
        args: list[str],
        cwd: Path | str | None,
        returncode: int | None,
        stderr: str,
    ) -> None:
        self.args = args
        self.cwd = cwd
        self.returncode = returncode
        self.stderr = stderr
        msg = f"git {' '.join(args)} (cwd={cwd}) exit={returncode}"
        if stderr:
            msg += f": {stderr.strip()[:500]}"
        super().__init__(msg)


@dataclass(frozen=True)
class GitResult:
    """Outcome of a successful git invocation."""

    stdout: str
    stderr: str


async def run_git(
    args: list[str],
    *,
    cwd: Path | str | None = None,
    env: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    check: bool = True,
) -> GitResult:
    """Run ``git <args>`` asynchronously and return decoded stdout/stderr.

    Args:
        args: argv passed after ``git`` (e.g. ``["worktree", "add", ...]``).
        cwd: Working directory; ``None`` uses the current process cwd.
        env: Subprocess environment; ``None`` inherits ``os.environ``.
        timeout: Hard wall-clock ceiling. Exceeded → SIGKILL +
            ``GitCommandError(returncode=None)``.
        check: If True (default), raise ``GitCommandError`` on non-zero
            exit. Set False to inspect failures explicitly.

    Returns:
        GitResult with decoded stdout / stderr.

    Raises:
        GitCommandError: On non-zero exit (when ``check=True``) or timeout.
    """
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=str(cwd) if cwd is not None else None,
        env=env if env is not None else os.environ.copy(),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        proc.kill()
        await proc.wait()
        raise GitCommandError(args, cwd, None, "timed out") from exc

    stdout = stdout_b.decode("utf-8", "replace") if stdout_b else ""
    stderr = stderr_b.decode("utf-8", "replace") if stderr_b else ""
    if check and proc.returncode != 0:
        raise GitCommandError(args, cwd, proc.returncode, stderr)
    return GitResult(stdout=stdout, stderr=stderr)
