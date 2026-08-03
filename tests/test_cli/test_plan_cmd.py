"""``molexp plan`` — CLI unit tests (no CliRunner, no app boot).

Function-level only: the integration tests that drove the Typer ``CliRunner``
and the server's UI-facing routes moved out of the pruned core. What remains
pins the approver seam — the CLI's :class:`~molexp.cli.plan_cmd.InteractiveApprover`
satisfies the harness ``Approver`` shape the emergent orchestrator expects — by
constructing both directly, with no CLI or FastAPI app boot.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace


class TestPlanCmd:
    def test_interactive_approver_is_a_valid_orchestrator_approve(self) -> None:
        """The CLI approver is an async ``(ApprovalRequest) -> ApprovalDecision``
        callable, and the emergent orchestrator accepts it as its ``approve`` seam."""
        from molexp.cli import plan_cmd
        from molexp.harness.modes.plan_orchestrator import PlanOrchestrator

        approver = plan_cmd.InteractiveApprover(
            run=SimpleNamespace(run_dir="/tmp/plan-run"), assume_yes=True
        )
        assert inspect.iscoroutinefunction(approver.__call__)
        # Constructs without error → the approver conforms to the Approver type.
        PlanOrchestrator(approve=approver)
