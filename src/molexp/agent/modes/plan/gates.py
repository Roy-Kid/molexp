"""Concrete approval gates shipped with PlanMode.

PlanMode binds the workflow-orthogonal :class:`~molexp.agent.policy.GatePolicy`
protocol to its own
:class:`~molexp.agent.modes.plan.schemas.PlanReviewView` /
:class:`~molexp.agent.modes.plan.schemas.ApprovalDecision` pair.
:class:`AutoApproveGatePolicy` (the safe non-interactive default) lives
at :mod:`molexp.agent.policy`; this module ships the concrete
*interactive* gates so callers do not re-implement them in every script.

The single gate today is :class:`PromptGatePolicy` — a CLI prompt that
walks the user through every generated task, surfaces the test +
implementation paths, and lets them allow / deny / peek before the
workflow advances. Modeled on Claude Code's "permission prompt"
vocabulary.
"""

from __future__ import annotations

from pathlib import Path

from molexp.agent.modes.plan.schemas import ApprovalDecision, PlanReviewView
from molexp.agent.policy import GatePolicy

__all__ = ["PromptGatePolicy"]


class PromptGatePolicy(GatePolicy[PlanReviewView, ApprovalDecision]):
    """Prompt the user task-by-task before approving the plan.

    On each ``human_review`` invocation:

    1. Prints the digest / plan-brief / validation summary so the user
       sees what the workflow as a whole agreed on.
    2. Iterates through every task in
       :attr:`PlanReviewView.contract.task_io` — for each, surfaces the
       generated test + implementation paths and prompts the user.
       ``y`` allows, ``n`` denies (replan), ``?`` shows the file
       contents inline so a decision does not need a second terminal.
    3. Returns one :class:`ApprovalDecision`. If every task was
       approved, ``approved=True``. Otherwise ``approved=False`` with
       ``target_task_ids`` listing the rejected ids and
       ``cascade_downstream=True`` so the repair loop reruns the
       validation chain over the regenerated artifacts.

    Freeform per-task notes the user types are concatenated into
    ``ApprovalDecision.feedback``; the repair loop persists them under
    ``repairs/iter-<n>/decision.yaml`` for audit.
    """

    async def human_review(self, view: PlanReviewView) -> ApprovalDecision:
        plan_root = view.experiment_workspace_path
        tests_dir = plan_root / "tests"
        impls_dir = plan_root / "src" / "experiment" / "tasks"

        print()
        print("=" * 72)
        print(f"PlanMode review — iteration {view.repair_iteration}")
        print("=" * 72)
        print(f"plan_id            : {view.plan_id}")
        print(f"workspace          : {plan_root}")
        print(f"validation passed  : {view.validation_passed}")
        print(f"validation summary : {view.validation_summary}")
        if view.previous_validation_failures:
            print("failed checks      :")
            for name in view.previous_validation_failures:
                print(f"  - {name}")
        print()
        print(f"experimental goal  : {view.digest.experimental_goal}")
        print(f"plan overview      : {view.plan_brief.overview}")
        if view.plan_brief.stages:
            print("plan stages        :")
            for stage in view.plan_brief.stages:
                print(f"  - {stage}")

        task_ios = view.contract.task_io
        if not task_ios:
            verdict = input("\nNo tasks in the contract. approve plan? [y/N]: ").strip().lower()
            return ApprovalDecision(
                approved=verdict in ("y", "yes"),
                reason="empty contract",
            )

        print()
        print(f"{len(task_ios)} task(s) to review one-by-one. "
              "y=approve, n=reject (replan), ?=show files.")

        rejected: list[str] = []
        feedback_lines: list[str] = []
        for idx, task_io in enumerate(task_ios, 1):
            task_id = task_io.task_id
            test_path = tests_dir / f"test_{task_id}.py"
            impl_path = impls_dir / f"{task_id}.py"
            print()
            print("-" * 72)
            print(f"Task {idx}/{len(task_ios)}: {task_id}")
            print("-" * 72)
            print(f"  test : {test_path}")
            print(f"  impl : {impl_path}")
            choice = _prompt_choice(
                "    approve this task? [y/N/?]: ",
                allow_show=True,
            )
            if choice == "?":
                _print_file(test_path, label="TEST")
                _print_file(impl_path, label="IMPL")
                choice = _prompt_choice(
                    "    approve this task? [y/N]: ",
                    allow_show=False,
                )
            if choice == "y":
                continue
            note = input("    what should the next round fix? (one line): ").strip()
            rejected.append(task_id)
            if note:
                feedback_lines.append(f"{task_id}: {note}")

        if not rejected:
            return ApprovalDecision(
                approved=True,
                reason=f"all {len(task_ios)} task(s) approved",
            )
        return ApprovalDecision(
            approved=False,
            reason=f"{len(rejected)}/{len(task_ios)} task(s) rejected",
            target_task_ids=tuple(rejected),
            cascade_downstream=True,
            feedback="\n".join(feedback_lines),
        )


def _prompt_choice(prompt: str, *, allow_show: bool) -> str:
    """Read one of {y, n, ?}; return the canonical letter (default ``n``)."""
    raw = input(prompt).strip().lower()
    if raw in ("y", "yes"):
        return "y"
    if allow_show and raw == "?":
        return "?"
    return "n"


def _print_file(path: Path, *, label: str) -> None:
    """Dump ``path`` to stdout with a banner; tolerate missing files."""
    print(f"  --- {label}: {path} ---")
    if not path.exists():
        print(f"  (missing: {path})")
        return
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        print(f"  | {line}")
    print(f"  --- end {label} ---")
