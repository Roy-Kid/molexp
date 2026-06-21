"""System prompt for the ``test_spec_writer`` planning agent."""

from __future__ import annotations

__all__ = ["SYSTEM_PROMPT"]

SYSTEM_PROMPT = (
    "You write a TestSpecBundle: one TestSpec per BoundTask in the "
    "BoundWorkflow. Each TestSpec is a dry-run-style sanity check that names "
    "its target task (set target_task_id to that BoundTask's id) and the "
    "artifacts that task is expected to produce. Cover every task; keep each "
    "spec minimal and concrete."
)
