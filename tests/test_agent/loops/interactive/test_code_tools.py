"""``code_tools`` — write_file / execute_python confinement and ExecutionEnv wiring.

Shared-path confinement (the ``..`` traversal branch) is owned by
``test_tools.py`` (the ``_safe_path`` helper); the agent→harness import
firewall is owned by ``tests/test_agent/test_import_guard.py``. This file
covers only what ``code_tools`` itself owns: the write byte-cap, absolute-path
handling, and every ``execute_python`` branch.
"""

from __future__ import annotations

import sys
from pathlib import Path

from molexp.agent.execution_env import ExecResult, ExecutionError
from molexp.agent.loops.interactive.code_tools import _MAX_WRITE_BYTES, code_tools


class FakeExecutionEnv:
    """Records ``exec`` calls; returns a fixed :class:`ExecResult`."""

    def __init__(self, scratch_dir: Path, *, result: ExecResult | None = None) -> None:
        self._scratch = Path(scratch_dir)
        self.calls: list[dict[str, object]] = []
        self.result = result or ExecResult(stdout="ok\n", stderr="", exit_code=0)
        self.raise_error: ExecutionError | None = None

    @property
    def scratch_dir(self) -> Path:
        self._scratch.mkdir(parents=True, exist_ok=True)
        return self._scratch

    def exec(
        self,
        command: list[str] | tuple[str, ...],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        self.calls.append(
            {
                "command": list(command),
                "cwd": Path(cwd).resolve() if cwd is not None else None,
                "timeout": timeout,
            }
        )
        if self.raise_error is not None:
            raise self.raise_error
        return self.result


def _tools(workspace: Path, execution_env: FakeExecutionEnv) -> dict[str, object]:
    return {
        tool.__name__: tool
        for tool in code_tools(workspace_root=workspace, execution_env=execution_env)
    }


class TestWriteFile:
    def test_writes_utf8_under_workspace(self, tmp_path: Path) -> None:
        write_file = _tools(tmp_path, FakeExecutionEnv(tmp_path / "s"))["write_file"]
        msg = write_file("notes/a.py", "x = 1\n")  # type: ignore[operator]
        target = tmp_path / "notes" / "a.py"
        assert target.is_file()
        assert target.read_text(encoding="utf-8") == "x = 1\n"
        assert "notes/a.py" in msg
        assert "bytes" in msg

    def test_allows_absolute_path_under_workspace(self, tmp_path: Path) -> None:
        write_file = _tools(tmp_path, FakeExecutionEnv(tmp_path / "s"))["write_file"]
        abs_path = str(tmp_path / "abs.py")
        msg = write_file(abs_path, "ok\n")  # type: ignore[operator]
        assert (tmp_path / "abs.py").read_text(encoding="utf-8") == "ok\n"
        assert "abs.py" in msg

    def test_rejects_absolute_path_outside_workspace(self, tmp_path: Path) -> None:
        write_file = _tools(tmp_path, FakeExecutionEnv(tmp_path / "s"))["write_file"]
        outside = tmp_path.parent / "outside-escape.py"
        out = write_file(str(outside), "nope")  # type: ignore[operator]
        assert out.startswith("error:")
        assert "escapes" in out
        assert not outside.exists()

    def test_rejects_oversized_content(self, tmp_path: Path) -> None:
        write_file = _tools(tmp_path, FakeExecutionEnv(tmp_path / "s"))["write_file"]
        payload = "x" * (_MAX_WRITE_BYTES + 1)
        out = write_file("big.py", payload)  # type: ignore[operator]
        assert out.startswith("error:")
        assert "refusing to write" in out
        assert not (tmp_path / "big.py").exists()


class TestExecutePython:
    def test_path_mode_runs_via_execution_env_with_workspace_cwd(self, tmp_path: Path) -> None:
        script = tmp_path / "scripts" / "yy.py"
        script.parent.mkdir()
        script.write_text("print('hello-yy')\n", encoding="utf-8")
        fake = FakeExecutionEnv(tmp_path / "scratch")
        execute_python = _tools(tmp_path, fake)["execute_python"]
        out = execute_python(path="scripts/yy.py")  # type: ignore[operator]
        assert len(fake.calls) == 1
        call = fake.calls[0]
        cmd = call["command"]
        assert isinstance(cmd, list)
        assert cmd[0] == sys.executable
        assert Path(cmd[1]) == script.resolve()
        assert call["cwd"] == tmp_path.resolve()
        assert "exit_code=0" in out
        assert "ok" in out

    def test_code_mode_writes_scratch_and_runs(self, tmp_path: Path) -> None:
        fake = FakeExecutionEnv(tmp_path / "scratch")
        execute_python = _tools(tmp_path, fake)["execute_python"]
        out = execute_python(code="print(1 + 1)")  # type: ignore[operator]
        assert len(fake.calls) == 1
        cmd = fake.calls[0]["command"]
        assert isinstance(cmd, list)
        script = Path(cmd[1])
        assert script.is_file()
        assert script.parent == fake.scratch_dir
        assert "print(1 + 1)" in script.read_text(encoding="utf-8")
        assert "exit_code=0" in out

    def test_requires_exactly_one_of_code_or_path(self, tmp_path: Path) -> None:
        execute_python = _tools(tmp_path, FakeExecutionEnv(tmp_path / "s"))["execute_python"]
        both = execute_python(code="print(1)", path="a.py")  # type: ignore[operator]
        neither = execute_python()  # type: ignore[operator]
        assert both.startswith("error:")
        assert neither.startswith("error:")

    def test_nonzero_exit_returned_as_text(self, tmp_path: Path) -> None:
        fake = FakeExecutionEnv(
            tmp_path / "s",
            result=ExecResult(stdout="", stderr="boom\n", exit_code=2),
        )
        execute_python = _tools(tmp_path, fake)["execute_python"]
        out = execute_python(code="raise SystemExit(2)")  # type: ignore[operator]
        assert "exit_code=2" in out
        assert "boom" in out

    def test_execution_error_surfaces_as_error_string(self, tmp_path: Path) -> None:
        fake = FakeExecutionEnv(tmp_path / "s")
        fake.raise_error = ExecutionError("timeout: command exceeded the 1.0s budget")
        execute_python = _tools(tmp_path, fake)["execute_python"]
        out = execute_python(code="pass", timeout=1.0)  # type: ignore[operator]
        assert out.startswith("error:")
        assert "timeout" in out
