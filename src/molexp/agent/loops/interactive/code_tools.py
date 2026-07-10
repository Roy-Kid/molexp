"""Write + execute tools for the emergent :class:`InteractiveLoop` loop.

Two bare callables — :func:`write_file`, :func:`execute_python` — always
mounted beside the read-only and knowledge tools. The model writes Python
under the workspace and runs it through the injected
:class:`~molexp.agent.execution_env.ExecutionEnv`; it does **not** get an
unrestricted shell.

Path confinement reuses :func:`~molexp.agent.loops.interactive.tools._safe_path`.
This module imports nothing from ``pydantic_ai`` or ``molexp.harness``.
"""

from __future__ import annotations

import sys
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from molexp.agent.execution_env import ExecutionError
from molexp.agent.loops.interactive.tools import _MAX_FILE_BYTES, _safe_path

if TYPE_CHECKING:
    from molexp.agent.execution_env import ExecutionEnv

__all__ = ["CODE_TOOL_NAMES", "code_tools"]

CODE_TOOL_NAMES = frozenset({"write_file", "execute_python"})

_MAX_WRITE_BYTES = _MAX_FILE_BYTES
"""Largest UTF-8 payload ``write_file`` will accept (bytes, same as read cap)."""

_MAX_EXEC_STREAM_CHARS = 32_768
"""Per-stream (stdout / stderr) character cap on ``execute_python`` results."""


def _truncate(text: str, *, limit: int = _MAX_EXEC_STREAM_CHARS) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n... ({omitted} more chars truncated)"


def code_tools(
    *,
    workspace_root: Path,
    execution_env: ExecutionEnv,
) -> tuple[Callable[..., str], ...]:
    """Build write/exec tool callables confined to ``workspace_root``.

    Args:
        workspace_root: Directory every path is confined to; also the
            default ``cwd`` for executed scripts.
        execution_env: Subprocess + scratch-dir boundary used by
            ``execute_python``.

    Returns:
        ``(write_file, execute_python)`` in a stable order.
    """
    root = Path(workspace_root).resolve()
    env = execution_env

    def write_file(path: str, content: str) -> str:
        """Write a UTF-8 text file under the workspace (creates parents).

        Args:
            path: Workspace-relative path, e.g. ``"scripts/yy.py"``.
            content: Full file body as a Unicode string. Rejected when
                UTF-8 encoding exceeds ``_MAX_WRITE_BYTES`` bytes.

        Returns:
            A short confirmation with the relative path and byte size.
        """
        try:
            raw = content.encode("utf-8")
            if len(raw) > _MAX_WRITE_BYTES:
                raise ValueError(
                    f"content is {len(raw)} bytes; refusing to write more than {_MAX_WRITE_BYTES}"
                )
            target = _safe_path(root, path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            rel = target.relative_to(root).as_posix()
            return f"wrote {rel} ({len(raw)} bytes)"
        except (ValueError, OSError) as exc:
            return f"error: {type(exc).__name__}: {exc}"

    def execute_python(
        code: str | None = None,
        path: str | None = None,
        timeout: float | None = None,
    ) -> str:
        """Run a Python script under the workspace via ExecutionEnv.

        Provide exactly one of ``path`` (workspace-relative script) or
        ``code`` (snippet written to the env scratch dir then executed).
        ``cwd`` is always the workspace root. Non-zero exit codes are
        returned as text (not raised); spawn failures / timeouts surface
        as an error string.

        Args:
            code: Inline Python source. Mutually exclusive with ``path``.
            path: Workspace-relative ``.py`` path. Mutually exclusive with
                ``code``.
            timeout: Optional hard timeout in seconds; defaults to the
                ExecutionEnv budget.

        Returns:
            A multi-line summary with ``exit_code``, truncated ``stdout``,
            and truncated ``stderr``.
        """
        if (code is None) == (path is None):
            return (
                "error: provide exactly one of code= or path= "
                "(inline snippet vs workspace-relative script)"
            )

        try:
            if path is not None:
                script = _safe_path(root, path)
                if not script.is_file():
                    return f"error: no such file in the workspace: {path!r}"
            else:
                assert code is not None
                snippet = code.encode("utf-8")
                if len(snippet) > _MAX_WRITE_BYTES:
                    return (
                        f"error: code is {len(snippet)} bytes; refusing more than "
                        f"{_MAX_WRITE_BYTES}"
                    )
                script = env.scratch_dir / f"agent_exec_{uuid.uuid4().hex[:12]}.py"
                script.write_bytes(snippet)

            result = env.exec(
                [sys.executable, str(script)],
                cwd=root,
                timeout=timeout,
            )
        except ExecutionError as exc:
            return f"error: {exc}"
        except (ValueError, OSError) as exc:
            return f"error: {type(exc).__name__}: {exc}"

        stdout = _truncate(result.stdout)
        stderr = _truncate(result.stderr)
        return f"exit_code={result.exit_code}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"

    write_file.__name__ = "write_file"
    execute_python.__name__ = "execute_python"
    return (write_file, execute_python)
