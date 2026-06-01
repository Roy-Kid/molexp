"""System prompt for the ``test_spec_writer`` planning agent."""

from __future__ import annotations

__all__ = ["SYSTEM_PROMPT"]

SYSTEM_PROMPT = (
    "You write a single TestSpec that sanity-checks the BoundWorkflow's first "
    "task: a dry-run-style check naming the target task and the artifacts it "
    "is expected to produce. Keep it minimal and concrete."
)
