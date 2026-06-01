"""The ``ExecutionEnv`` subprocess + scratch-dir abstraction.

:class:`ExecutionEnv` is the shell/subprocess boundary the harness
exposes to modes that need to run a command (e.g. a generated test
suite). It is *only* a process-execution abstraction plus one scratch
directory — it does **not** replace :mod:`molexp.workspace` storage.

Two concrete classes:

- :class:`LocalExecutionEnv` — runs real subprocesses with a hard
  timeout, confined working directory, and a lazily-created scratch
  dir. The production implementation.
- a ``FakeExecutionEnv`` for unit tests lives under
  ``tests/test_agent/harness/conftest.py``, not here.

Container / sandboxed implementations are intentionally deferred — the
Protocol leaves that story open for a future spec.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

__all__ = [
    "ExecResult",
    "ExecutionEnv",
    "ExecutionError",
    "LocalExecutionEnv",
]

# Default hard ceiling so a runaway subprocess can never wedge a run.
_DEFAULT_TIMEOUT_SECONDS = 120.0


class ExecutionError(RuntimeError):
    """Raised when a subprocess fails to spawn or exceeds its timeout.

    A non-zero *exit code* is **not** an error — that is a normal
    :class:`ExecResult` outcome. ``ExecutionError`` is reserved for
    spawn failures (binary not found, permission denied) and timeouts.
    """


class ExecResult(BaseModel):
    """The captured outcome of one :meth:`ExecutionEnv.exec` call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stdout: str
    stderr: str
    exit_code: int


@runtime_checkable
class ExecutionEnv(Protocol):
    """Process-execution abstraction — a command runner + one scratch dir."""

    @property
    def scratch_dir(self) -> Path:
        """A confined directory the env owns for transient files."""
        ...

    def exec(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        """Run ``command`` and return its captured :class:`ExecResult`.

        Raises:
            ExecutionError: on spawn failure or timeout.
        """
        ...


class LocalExecutionEnv:
    """Run subprocesses locally with a hard timeout and confined ``cwd``.

    Plain runtime class — it owns a scratch directory and shells out.
    The scratch dir is created lazily on first :attr:`scratch_dir`
    access (or first :meth:`exec` that defaults its ``cwd`` to it).
    """

    def __init__(
        self,
        *,
        scratch_dir: Path | str,
        default_timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._scratch_dir = Path(scratch_dir).resolve()
        self._default_timeout = default_timeout

    @property
    def scratch_dir(self) -> Path:
        """The env's confined scratch directory, created on first access."""
        self._scratch_dir.mkdir(parents=True, exist_ok=True)
        return self._scratch_dir

    def exec(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        """Run ``command`` via :mod:`subprocess` with a hard timeout.

        ``cwd`` defaults to :attr:`scratch_dir`; ``timeout`` defaults to
        the env's ``default_timeout``. ``env``, when given, is the
        *complete* environment for the child (not merged with the
        parent's).
        """
        workdir = Path(cwd).resolve() if cwd is not None else self.scratch_dir
        effective_timeout = timeout if timeout is not None else self._default_timeout
        run_env = dict(env) if env is not None else None
        try:
            completed = subprocess.run(
                list(command),
                cwd=str(workdir),
                env=run_env,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ExecutionError(
                f"timeout: command exceeded the {effective_timeout}s budget: {list(command)!r}"
            ) from exc
        except (OSError, ValueError) as exc:
            raise ExecutionError(f"failed to spawn command {list(command)!r}: {exc}") from exc
        return ExecResult(
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            exit_code=completed.returncode,
        )
