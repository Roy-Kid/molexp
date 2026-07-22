"""``molexp plan`` — CLI unit tests (no CliRunner, no app boot).

Function-level only: the integration tests that drove the Typer ``CliRunner``
and the server's UI-facing routes moved out of the pruned core. What remains
pins the executor seam default — ``PlanRuntime.build_executor()`` returns a
:class:`~molexp.harness.LocalExecutor` — asserted by calling the factory
directly, with no CLI or FastAPI app boot.
"""

from __future__ import annotations


class TestPlanCmd:
    def test_executor_seam_defaults_to_local_executor(self) -> None:
        from molexp.cli import plan_cmd
        from molexp.harness import LocalExecutor

        assert isinstance(plan_cmd.PlanRuntime.build_executor(), LocalExecutor)
