"""Subprocess sandbox for agent-authored Task code.

Agent-authored Python files (e.g. landed by an external coding agent into
``workspace_root/.scratch/agent_tasks/``) execute via
:func:`run_in_sandbox` so the host process is shielded from misbehaving
or malicious user code.

The sandbox is **not** a security boundary in the OS sense — it is a
process boundary plus a filesystem-scope contract.  A motivated attacker
who controls the agent's Python source can still misuse OS facilities;
the goal here is to:

1. Quarantine the agent's CPU + memory inside a child process so a stuck
   loop or memory leak does not take down the orchestrator.
2. Pin the working directory so relative paths land where the host
   expects.
3. Reject writes / reads that escape an opt-in allow-list of filesystem
   roots before exec, by inserting a tiny instrumentation shim in front
   of the agent script.

Stronger isolation (seccomp, namespaces, containerisation) is a separate
deployment concern outside this module's contract.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class SandboxError(RuntimeError):
    """Base class for sandbox-side failures."""


class SandboxPermissionError(SandboxError):
    """Sandboxed script attempted a filesystem operation outside the allow-list."""


class SandboxTimeoutError(SandboxError):
    """Sandboxed script exceeded the configured wall-clock timeout."""


class SandboxResult(BaseModel):
    """Outcome of a single :func:`run_in_sandbox` invocation."""

    model_config = ConfigDict(frozen=True)

    returncode: int
    stdout: str
    stderr: str


def run_in_sandbox(
    script: Path,
    *,
    cwd: Path,
    timeout: float | None = None,
    allowed_read_roots: tuple[Path, ...] = (),
    allowed_write_roots: tuple[Path, ...] = (),
    env: dict[str, str] | None = None,
) -> SandboxResult:
    """Execute ``script`` in a Python subprocess pinned to ``cwd``.

    Args:
        script: Path to the agent-authored Python file to run.
        cwd: Working directory the subprocess starts in.  Always resolved
            to an absolute path before the script runs.
        timeout: Wall-clock seconds before the subprocess is killed and
            :class:`SandboxTimeoutError` is raised.  ``None`` waits
            indefinitely.
        allowed_read_roots: Filesystem prefixes the script may read from.
            Defaults to ``(cwd,)`` when empty.  Any other ``open(..., 'r')``
            target raises :class:`SandboxPermissionError` *inside the
            subprocess*, which the caller observes as a non-zero return
            code plus the exception text on stderr.
        allowed_write_roots: Same as ``allowed_read_roots`` for write
            modes.  Defaults to ``(cwd,)`` when empty.
        env: Environment variables for the subprocess.  ``None`` inherits
            the current environment.

    Returns:
        A :class:`SandboxResult` describing the subprocess outcome.

    Raises:
        SandboxTimeoutError: ``timeout`` was reached before the script
            finished.
        SandboxPermissionError: The script executed an out-of-scope
            filesystem operation.  Detected by inspecting the
            subprocess's stderr after exit.
    """
    cwd_resolved = Path(cwd).resolve()
    if not cwd_resolved.is_dir():
        raise SandboxError(f"sandbox cwd {cwd_resolved} is not a directory")

    read_roots = tuple(p.resolve() for p in (allowed_read_roots or (cwd_resolved,)))
    write_roots = tuple(p.resolve() for p in (allowed_write_roots or (cwd_resolved,)))

    shim = _build_shim(script.resolve(), read_roots, write_roots)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=str(cwd_resolved), delete=False
    ) as tmp:
        tmp.write(shim)
        shim_path = tmp.name

    try:
        try:
            completed = subprocess.run(
                [sys.executable, shim_path],
                cwd=str(cwd_resolved),
                env={**os.environ, **(env or {})},
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SandboxTimeoutError(f"sandbox script exceeded timeout={timeout}s") from exc
    finally:
        with contextlib.suppress(OSError):
            os.unlink(shim_path)

    if "SandboxPermissionError:" in completed.stderr:
        raise SandboxPermissionError(completed.stderr.strip())

    return SandboxResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _build_shim(
    script_path: Path,
    read_roots: tuple[Path, ...],
    write_roots: tuple[Path, ...],
) -> str:
    """Return Python source that wraps ``open`` and ``os.open`` with allow-list checks.

    The shim is executed in the subprocess and patches the builtins
    *before* importing the user's script.  Any out-of-scope path raises
    ``SandboxPermissionError`` on stderr which the parent observes.
    """
    return textwrap.dedent(
        """
        import builtins as _b
        import os as _os
        import runpy as _rp
        import sys as _sys

        _SCRIPT = {script!r}
        _READ = {read!r}
        _WRITE = {write!r}

        _real_open = _b.open
        _real_os_open = _os.open

        def _is_under(target, roots):
            try:
                resolved = _os.path.realpath(target)
            except (TypeError, ValueError, OSError):
                return False
            return any(
                resolved == root or resolved.startswith(root + _os.sep)
                for root in roots
            )

        def _check(path, mode):
            mode_str = mode if isinstance(mode, str) else "r"
            wants_write = any(c in mode_str for c in "wax+")
            roots = _WRITE if wants_write else _READ
            if not _is_under(str(path), roots):
                _sys.stderr.write(
                    "SandboxPermissionError: "
                    + repr(str(path))
                    + " is outside allowed roots "
                    + repr(roots)
                    + "\\n"
                )
                raise PermissionError(str(path))

        def _open(file, mode="r", *args, **kwargs):
            _check(file, mode)
            return _real_open(file, mode, *args, **kwargs)

        def _os_open(path, flags, *args, **kwargs):
            wants_write = bool(flags & (_os.O_WRONLY | _os.O_RDWR | _os.O_CREAT))
            mode = "w" if wants_write else "r"
            _check(path, mode)
            return _real_os_open(path, flags, *args, **kwargs)

        _b.open = _open
        _os.open = _os_open

        _rp.run_path(_SCRIPT, run_name="__main__")
        """
    ).format(
        script=str(script_path),
        read=tuple(str(p) for p in read_roots),
        write=tuple(str(p) for p in write_roots),
    )


__all__ = [
    "SandboxError",
    "SandboxPermissionError",
    "SandboxResult",
    "SandboxTimeoutError",
    "run_in_sandbox",
]
