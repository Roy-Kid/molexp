"""RED tests for molexp-research-orchestration agent-layer extensions.

Covers:

- ``molexp.agent.Agent(provider=...)`` unified provider factory
  (acceptance ac-006 supporting infrastructure).
- ``molexp.agent.replan(run, modifier)`` replanned-from sibling Run
  (acceptance ac-006 / ac-007).
- ``molexp.agent.sandbox.run_in_sandbox()`` subprocess sandbox for
  agent-authored Task code (acceptance ac-008).

Workflow integration is delegated to the integration chunk; this file
exercises the agent-layer primitives in isolation.
"""

from __future__ import annotations

import textwrap

import pytest

from molexp.workspace import Workspace

# ── Agent(provider=...) factory ──────────────────────────────────────────────


class TestAgentFactory:
    def test_unknown_provider_raises(self) -> None:
        from molexp.agent import Agent

        with pytest.raises(ValueError):
            Agent(provider="not-a-real-provider")

    def test_known_provider_returns_client_class(self) -> None:
        """Smoke test — factory should resolve the configured provider class
        without instantiating it (instantiation often needs a CLI binary)."""
        from molexp.agent import Agent

        # ``resolve_only=True`` returns the class without spawning a subprocess.
        cls = Agent(provider="claude", resolve_only=True)
        assert cls.__name__ == "ClaudeCliClient"

        cls2 = Agent(provider="codex", resolve_only=True)
        assert cls2.__name__ == "CodexAppServerClient"

    def test_list_providers_exposes_known_names(self) -> None:
        from molexp.agent import Agent

        names = Agent.list_providers()
        assert "claude" in names
        assert "codex" in names


# ── replan ──────────────────────────────────────────────────────────────────


class TestReplan:
    def test_replan_creates_sibling_run_in_same_experiment(self, tmp_path) -> None:
        from molexp.agent.replan import replan

        ws = Workspace(root=tmp_path, name="replan-test")
        proj = ws.project("p1")
        exp = proj.experiment("e1")
        original = exp.run(parameters={"charge_strength": 1.0})
        with original.start() as ctx:
            ctx.set_result("collapsed", True)

        new_run = replan(
            original,
            modifier=lambda params: {**params, "charge_strength": params["charge_strength"] / 2},
            reason="test sanity miss",
        )

        # New run is in the same experiment.
        assert new_run.experiment.id == exp.id
        # Parameters reflect the modifier output.
        assert new_run.parameters["charge_strength"] == 0.5
        # Provenance recorded.
        assert new_run.metadata.labels.get("replanned_from") == original.id
        assert new_run.metadata.labels.get("replanned_reason") == "test sanity miss"
        # Original is untouched.
        assert original.parameters["charge_strength"] == 1.0

    def test_replan_modifier_can_return_dict_directly(self, tmp_path) -> None:
        from molexp.agent.replan import replan

        ws = Workspace(root=tmp_path, name="replan-direct")
        exp = ws.project("p1").experiment("e1")
        original = exp.run(parameters={"a": 1})

        new_run = replan(original, modifier=lambda params: {"a": 99})
        assert new_run.parameters["a"] == 99


# ── sandbox ─────────────────────────────────────────────────────────────────


class TestSandbox:
    def test_runs_a_simple_script_and_returns_stdout(self, tmp_path) -> None:
        from molexp.agent.sandbox import run_in_sandbox

        script = tmp_path / "hello.py"
        script.write_text("print('hello sandbox')")

        result = run_in_sandbox(script, cwd=tmp_path)
        assert result.returncode == 0
        assert "hello sandbox" in result.stdout

    def test_sandbox_cwd_pins_working_directory(self, tmp_path) -> None:
        from molexp.agent.sandbox import run_in_sandbox

        script = tmp_path / "pwd.py"
        script.write_text(
            textwrap.dedent(
                """
                import os
                print(os.getcwd())
                """
            )
        )
        result = run_in_sandbox(script, cwd=tmp_path)
        assert result.returncode == 0
        # cwd output should be the directory we pinned.
        assert str(tmp_path.resolve()) in result.stdout

    def test_sandbox_blocks_path_traversal_outside_cwd(self, tmp_path) -> None:
        """Reading a file outside ``cwd`` raises a sandbox PermissionError.

        The sandbox enforces a fs-scope contract via a chdir + path-prefix
        check at exec time. Direct subprocess sandboxing of arbitrary fs
        access is OS-specific; we encode the policy at the sandbox helper
        boundary by rejecting the run when the script reads outside cwd.
        """
        from molexp.agent.sandbox import SandboxPermissionError, run_in_sandbox

        outside_target = tmp_path.parent / "should_not_read.txt"
        outside_target.write_text("secret")
        try:
            script = tmp_path / "escape.py"
            script.write_text(
                textwrap.dedent(
                    f"""
                    import sys
                    with open({str(outside_target)!r}) as fh:
                        sys.stdout.write(fh.read())
                    """
                )
            )
            with pytest.raises(SandboxPermissionError):
                run_in_sandbox(
                    script,
                    cwd=tmp_path,
                    allowed_read_roots=(tmp_path,),
                )
        finally:
            outside_target.unlink()

    def test_sandbox_timeout_raises(self, tmp_path) -> None:
        from molexp.agent.sandbox import SandboxTimeoutError, run_in_sandbox

        script = tmp_path / "loop.py"
        script.write_text(
            textwrap.dedent(
                """
                import time
                while True:
                    time.sleep(0.1)
                """
            )
        )
        with pytest.raises(SandboxTimeoutError):
            run_in_sandbox(script, cwd=tmp_path, timeout=0.5)
