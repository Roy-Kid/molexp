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


#: Chat Mode default write/run confinement (under workspace root).
CHAT_SCRATCH_PREFIX = "agent/.scratch"


class LocalCodeEnv:
    """Workspace-confined write + Python exec via :class:`ExecutionEnv`."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        execution_env: ExecutionEnv,
        confine_to: str | None = None,
    ) -> None:
        self._root = Path(workspace_root).resolve()
        self._env = execution_env
        # When set (e.g. ``agent/.scratch``), all writes and path= runs must
        # resolve under that prefix — Chat Mode default.
        self._confine = confine_to.strip().strip("/") if confine_to else None

    def _confine_rel(self, path: str) -> str:
        """Map *path* into the confinement prefix when active."""
        rel = path.replace("\\", "/").lstrip("./")
        if not self._confine:
            return rel
        prefix = self._confine
        if rel == prefix or rel.startswith(f"{prefix}/"):
            return rel
        # Absolute-looking or projects/… escapes → force basename under scratch.
        name = Path(rel).name or "script.py"
        return f"{prefix}/{name}" if not rel or ".." in Path(rel).parts else f"{prefix}/{rel}"

    def write(self, path: str, content: str) -> WriteResult:
        try:
            raw = content.encode("utf-8")
            if len(raw) > _MAX_WRITE_BYTES:
                raise ValueError(
                    f"content is {len(raw)} bytes; refusing more than {_MAX_WRITE_BYTES}"
                )
            rel = self._confine_rel(path)
            target = safe_path(self._root, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            out = target.relative_to(self._root).as_posix()
            return WriteResult(path=out, bytes_written=len(raw))
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
                rel = self._confine_rel(path)
                script = safe_path(self._root, rel)
                if not script.is_file():
                    # Also try original path if confinement remapped and file
                    # already existed under scratch with user path.
                    if not script.is_file() and self._confine:
                        alt = safe_path(self._root, path)
                        if alt.is_file() and str(alt.resolve()).startswith(
                            str((self._root / self._confine).resolve())
                        ):
                            script = alt
                    if not script.is_file():
                        return RunResult(
                            exit_code=-1,
                            ok=False,
                            error=f"no such file in the workspace: {rel!r}",
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
                scratch = self._root / self._confine if self._confine else self._env.scratch_dir
                scratch.mkdir(parents=True, exist_ok=True)
                script = scratch / f"agent_exec_{uuid.uuid4().hex[:12]}.py"
                script.write_bytes(snippet)
            # Chat Mode: run with cwd under scratch so relative open()/Path
            # writes cannot invent projects/ by accident. Scripts that need
            # the workspace root must use an absolute path (still allowed for
            # read/science libs); structure mutators are denied via tools +
            # preamble, not by blocking Python imports wholesale.
            run_cwd = (self._root / self._confine) if self._confine else self._root
            if self._confine:
                run_cwd.mkdir(parents=True, exist_ok=True)
            result = self._env.exec(
                [sys.executable, str(script)],
                cwd=run_cwd,
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
