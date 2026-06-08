"""CommandTask — run one external command, check its return code, fail on nonzero.

A reusable batch :class:`~molexp.workflow.task.Task` that collapses the
recurring "run a command, check ``returncode``, raise on non-zero" wrapper
block into one primitive. A non-zero exit surfaces as a
:class:`~molexp.workflow.types.CommandError` (a ``WorkflowError``) carrying the
command's ``returncode`` / ``stdout`` / ``stderr``.

Two construction paths, exactly one supplied:

* **argv** — ``CommandTask(["antechamber", "-i", ...])`` runs the list as a
  subprocess via :func:`subprocess.run` (capturing stdout/stderr; honoring an
  optional ``cwd`` / ``env`` / ``timeout``).
* **runner** — ``CommandTask(runner=fn)`` calls a zero-arg callable returning
  any object exposing ``.returncode`` / ``.stdout`` / ``.stderr``. This lets
  callers (e.g. molpy wrapper tasks) reuse ``CommandTask`` without molexp
  depending on molpy — the wrapper is invoked in the caller's closure and
  ``CommandTask`` only reads the three attributes off the result.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from molexp.workflow.task import Task
from molexp.workflow.types import CommandError

if TYPE_CHECKING:
    from molexp.workflow.context import TaskContext


def _as_str(value: Any) -> str:  # noqa: ANN401 — duck-typed stream attr (str | bytes | None)
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value if isinstance(value, str) else str(value)


@dataclass(frozen=True)
class CommandResult:
    """Normalized outcome of a command: its return code and captured streams."""

    returncode: int
    stdout: str
    stderr: str


class CommandTask(Task):
    """Run one external command and fail the workflow on a non-zero return code.

    Args:
        argv: The command + arguments to run as a subprocess. Mutually exclusive
            with *runner*.
        runner: A zero-arg callable returning an object with ``.returncode`` /
            ``.stdout`` / ``.stderr``. Mutually exclusive with *argv*.
        cwd: Working directory for the argv subprocess (ignored for *runner*).
        env: Environment mapping for the argv subprocess (ignored for *runner*).
        timeout: Optional wall-clock timeout (seconds) for the argv subprocess.

    Raises:
        ValueError: If both or neither of *argv* / *runner* are supplied.
    """

    def __init__(
        self,
        argv: Sequence[str] | None = None,
        *,
        runner: Callable[[], Any] | None = None,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> None:
        if (argv is None) == (runner is None):
            raise ValueError("CommandTask requires exactly one of `argv` or `runner`")
        self._argv = list(argv) if argv is not None else None
        self._runner = runner
        self._cwd = cwd
        self._env = dict(env) if env is not None else None
        self._timeout = timeout

    async def execute(self, ctx: TaskContext) -> CommandResult:  # noqa: ARG002 — argv/runner ignore ctx
        """Run the command and return its :class:`CommandResult`.

        Returns:
            The normalized result on a zero return code.

        Raises:
            CommandError: If the command exits with a non-zero return code; the
                error carries ``returncode`` / ``stdout`` / ``stderr``.
        """
        if self._runner is not None:
            raw = self._runner()
            result = CommandResult(int(raw.returncode), _as_str(raw.stdout), _as_str(raw.stderr))
        else:
            completed = subprocess.run(
                self._argv if self._argv is not None else [],
                capture_output=True,
                text=True,
                cwd=self._cwd,
                env=self._env,
                timeout=self._timeout,
                check=False,
            )
            result = CommandResult(
                completed.returncode, _as_str(completed.stdout), _as_str(completed.stderr)
            )
        if result.returncode != 0:
            raise CommandError(result.returncode, result.stdout, result.stderr)
        return result
