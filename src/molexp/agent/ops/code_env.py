"""Local :class:`~molexp.agent.ops.protocols.CodeEnv` over ExecutionEnv."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from molexp.agent.execution_env import ExecutionError
from molexp.agent.ops.paths import safe_path
from molexp.agent.ops.protocols import RunResult, WriteResult

if TYPE_CHECKING:
    from molexp.agent.execution_env import ExecutionEnv

_MAX_WRITE_BYTES = 512 * 1024
_MAX_STREAM_CHARS = 32_768


def _truncate(text: str, *, limit: int = _MAX_STREAM_CHARS) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... ({len(text) - limit} more chars truncated)"


class LocalCodeEnv:
    """Workspace-confined write + Python exec via :class:`ExecutionEnv`."""

    def __init__(self, *, workspace_root: Path, execution_env: ExecutionEnv) -> None:
        self._root = Path(workspace_root).resolve()
        self._env = execution_env

    def write(self, path: str, content: str) -> WriteResult:
        try:
            raw = content.encode("utf-8")
            if len(raw) > _MAX_WRITE_BYTES:
                raise ValueError(
                    f"content is {len(raw)} bytes; refusing more than {_MAX_WRITE_BYTES}"
                )
            target = safe_path(self._root, path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            rel = target.relative_to(self._root).as_posix()
            return WriteResult(path=rel, bytes_written=len(raw))
        except (ValueError, OSError) as exc:
            return WriteResult(path=path, bytes_written=0, ok=False, error=str(exc))

    def run(
        self,
        *,
        path: str | None = None,
        code: str | None = None,
        timeout: float | None = None,
    ) -> RunResult:
        if (code is None) == (path is None):
            return RunResult(
                exit_code=-1,
                ok=False,
                error="provide exactly one of code= or path=",
            )
        try:
            if path is not None:
                script = safe_path(self._root, path)
                if not script.is_file():
                    return RunResult(
                        exit_code=-1,
                        ok=False,
                        error=f"no such file in the workspace: {path!r}",
                    )
            else:
                assert code is not None
                snippet = code.encode("utf-8")
                if len(snippet) > _MAX_WRITE_BYTES:
                    return RunResult(
                        exit_code=-1,
                        ok=False,
                        error=f"code is {len(snippet)} bytes; refusing more than {_MAX_WRITE_BYTES}",
                    )
                script = self._env.scratch_dir / f"agent_exec_{uuid.uuid4().hex[:12]}.py"
                script.write_bytes(snippet)
            result = self._env.exec(
                [sys.executable, str(script)],
                cwd=self._root,
                timeout=timeout,
            )
        except ExecutionError as exc:
            return RunResult(exit_code=-1, ok=False, error=str(exc))
        except (ValueError, OSError) as exc:
            return RunResult(exit_code=-1, ok=False, error=str(exc))
        return RunResult(
            exit_code=result.exit_code,
            stdout=_truncate(result.stdout),
            stderr=_truncate(result.stderr),
            ok=result.exit_code == 0,
        )
