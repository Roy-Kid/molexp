"""Runtime context and progress rendering for PlanMode.

This module is intentionally small and dependency-light.  It owns the
cross-cutting runtime concerns that should not be reimplemented in each
workflow node:

* repair context threaded from a review / capability-gate rejection into
  the next LLM round;
* human-sized progress labels for the 13-node plan pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from molexp.agent.review import ReviewDecision

__all__ = [
    "PLAN_PIPELINE_ORDER",
    "PlanRepairContext",
    "format_node_label",
    "format_progress_done",
    "format_progress_start",
]


PLAN_PIPELINE_ORDER: tuple[str, ...] = (
    "IngestReport",
    "DraftReportDigest",
    "DraftImplementationPlan",
    "DraftCapabilityNeeds",
    "DiscoverCapabilities",
    "CompileWorkflowIR",
    "CompileTaskIR",
    "GenerateWorkflowSkeleton",
    "GenerateTaskTests",
    "GenerateTaskImplementations",
    "ValidateWorkspace",
    "HumanReview",
    "FinalHandoffCheck",
)
"""Stable topological order of the PlanMode workflow."""


_PIPELINE_TOTAL = len(PLAN_PIPELINE_ORDER)
_PIPELINE_INDEX: dict[str, int] = {name: i for i, name in enumerate(PLAN_PIPELINE_ORDER, 1)}


@dataclass(frozen=True)
class PlanRepairContext:
    """Structured instruction carried into a repair iteration.

    The repair loop builds this from the exact :class:`ReviewDecision`
    that caused a re-run.  LLM-facing nodes receive it through
    :class:`~molexp.agent.modes.plan.protocols.PlanDeps` and append the
    rendered block to their prompts centrally, so feedback is not lost
    between review and regeneration.
    """

    iteration: int = 0
    source: str = ""
    reason: str = ""
    feedback: str = ""
    target_steps: tuple[str, ...] = ()
    target_task_ids: tuple[str, ...] = ()
    cascade_downstream: bool = False

    @classmethod
    def from_decision(
        cls,
        *,
        iteration: int,
        decision: ReviewDecision,
        source: str,
    ) -> PlanRepairContext:
        """Create context for the next repair round."""
        return cls(
            iteration=iteration,
            source=source,
            reason=decision.reason,
            feedback=decision.feedback,
            target_steps=decision.target_steps,
            target_task_ids=decision.target_task_ids,
            cascade_downstream=decision.cascade_downstream,
        )

    @property
    def active(self) -> bool:
        """Whether this run is a repair attempt rather than the first pass."""
        return bool(
            self.iteration
            or self.reason
            or self.feedback
            or self.target_steps
            or self.target_task_ids
        )

    def prompt_block(self, *, node_id: str = "", task_id: str = "") -> str:
        """Render a compact, binding prompt appendix for an LLM call."""
        if not self.active:
            return ""

        lines = [
            "## Repair context (binding)",
            "",
            f"This is repair iteration {self.iteration}. Address the previous rejection",
            "directly while preserving already-approved behavior unless the targeted",
            "step/task or downstream validation requires a change.",
        ]
        if node_id:
            lines.append(f"Current plan node: {node_id}.")
        if task_id:
            lines.append(f"Current experiment task: {task_id}.")
        if self.source:
            lines.append(f"Rejection source: {self.source}.")
        if self.reason:
            lines.append(f"Reason: {self.reason}.")
        if self.feedback:
            lines.append("Reviewer feedback:")
            lines.append(self.feedback)
        if self.target_steps:
            lines.append(f"Target plan steps: {', '.join(self.target_steps)}.")
        if self.target_task_ids:
            lines.append(f"Target experiment tasks: {', '.join(self.target_task_ids)}.")
        if self.cascade_downstream:
            lines.append(
                "Downstream artifacts are being regenerated because the rejection cascades."
            )
        lines.append(
            "If this feedback conflicts with capability evidence, workflow schema, or validation "
            "rules, follow the schema/evidence and make the smallest defensible change."
        )
        return "\n".join(lines)

    def append_to_prompt(self, user: str, *, node_id: str = "", task_id: str = "") -> str:
        """Append the repair block to a user prompt when active."""
        block = self.prompt_block(node_id=node_id, task_id=task_id)
        if not block:
            return user
        return f"{user}\n\n{block}"


def format_node_label(node_name: str) -> str:
    """Human-readable node label, e.g. ``03/13 DraftImplementationPlan``."""
    idx = _PIPELINE_INDEX.get(node_name)
    if idx is None:
        return f"--/{_PIPELINE_TOTAL:02d} {node_name}"
    return f"{idx:02d}/{_PIPELINE_TOTAL:02d} {node_name}"


def format_progress_start(node_name: str, *, iteration: int = 0) -> str:
    """Progress line for node start."""
    if iteration:
        return f"[plan] {format_node_label(node_name)} start (repair {iteration})"
    return f"[plan] {format_node_label(node_name)} start"


def format_progress_done(node_name: str, summary: str = "") -> str:
    """Progress line for successful node completion."""
    suffix = f" - {summary}" if summary else ""
    return f"[plan] {format_node_label(node_name)} done{suffix}"
