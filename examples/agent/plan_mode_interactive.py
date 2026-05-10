"""Interactive driver for ``PlanMode`` with the review→repair loop.

Demonstrates a stdin-driven :class:`StdinInteractiveGatePolicy` that
prompts the operator after each iteration to either approve the
materialized plan or request a partial-rerun. Companion to the
``planmode-review-repair-loop`` spec; the policy emits an
:class:`ApprovalDecision` carrying ``target_node_ids`` /
``target_task_ids`` / ``cascade_downstream`` / ``feedback`` so the
outer driver can shape the next iteration's
:meth:`Workflow.subgraph` selection.

Run directly::

    python examples/agent/plan_mode_interactive.py < input.txt

(or interactively in a terminal). Set ``OPENAI_API_KEY`` and pass a
real model id in :data:`MODEL_ID` to drive a non-test provider.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from molexp.agent import AgentRunner, AgentSession
from molexp.agent.modes import PlanMode
from molexp.agent.modes.plan import (
    GatePolicy,
    PlanWorkspaceHandle,
)
from molexp.agent.modes.plan.schemas import ApprovalDecision, PlanReviewView
from molexp.workspace import Workspace

MODEL_ID = "test"
"""pydantic-ai model id; replace with e.g. ``"openai:gpt-4o-mini"`` for real
LLM dispatch (requires ``OPENAI_API_KEY`` in the environment)."""


REPORT = (
    "Investigate Suzuki coupling: maximize biaryl yield over a Pd loading × "
    "temperature grid using a fixed solvent / base set."
)


class StdinInteractiveGatePolicy(GatePolicy):
    """Approve / reject prompts threaded through ``input()``.

    On each call the policy prints a brief summary of the
    :class:`PlanReviewView`, asks for ``approve / reject`` and — on
    rejection — collects the structured repair targets. Designed for
    CLI demos and integration tests; production UIs should implement
    :class:`GatePolicy` against a richer transport (WebSocket, IPC,
    REST, …).
    """

    async def human_review(self, view: PlanReviewView) -> ApprovalDecision:
        print()
        print("=" * 64)
        print(f"PlanMode review — iteration {view.repair_iteration}")
        print("=" * 64)
        print(f"plan_id            : {view.plan_id}")
        print(f"workspace          : {view.experiment_workspace_path}")
        print(f"validation passed  : {view.validation_passed}")
        print(f"validation summary : {view.validation_summary}")
        if view.previous_validation_failures:
            print("failed checks      :")
            for name in view.previous_validation_failures:
                print(f"  - {name}")
        print(f"experimental goal  : {view.digest.experimental_goal}")
        print(f"plan overview      : {view.plan_brief.overview}")

        verdict = input("approve? [y/N]: ").strip().lower()
        if verdict in ("y", "yes"):
            return ApprovalDecision(approved=True, reason="operator approved")

        print()
        print("Rejection — describe the repair scope.")
        node_csv = input(
            "  target plan-level nodes (comma-separated, blank for none): "
        ).strip()
        task_csv = input(
            "  target experiment-task ids   (comma-separated, blank for none): "
        ).strip()
        cascade_raw = input("  cascade downstream? [y/N]: ").strip().lower()
        feedback = input("  freeform feedback (one line): ").strip()

        return ApprovalDecision(
            approved=False,
            reason="operator rejected",
            target_node_ids=tuple(_csv(node_csv)),
            target_task_ids=tuple(_csv(task_csv)),
            cascade_downstream=cascade_raw in ("y", "yes"),
            feedback=feedback,
        )


def _csv(raw: str) -> list[str]:
    """Parse a comma-separated list, stripping blanks."""
    return [item.strip() for item in raw.split(",") if item.strip()]


async def main() -> int:
    with TemporaryDirectory() as tmp:
        workspace = Workspace(Path(tmp) / "ws")
        handle = PlanWorkspaceHandle.materialize(workspace, plan_id="interactive_demo")
        mode = PlanMode(
            workspace_handle=handle,
            gate_policy=StdinInteractiveGatePolicy(),
            max_iterations=4,
        )
        runner = AgentRunner(mode=mode, model=MODEL_ID)
        session = AgentSession()
        result = await runner.run(session, REPORT)

        print()
        print("=" * 64)
        print("Run finished")
        print("=" * 64)
        print(result.text)
        plan = (result.mode_state or {}).get("plan", {})
        print(f"approved      : {plan.get('approved')}")
        print(f"ready_for_run : {plan.get('ready_for_run')}")
        print(f"status        : {plan.get('status')}")
        print(f"workspace at  : {handle.root()}")
        print(f"manifest at   : {handle.manifest_path()}")
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
