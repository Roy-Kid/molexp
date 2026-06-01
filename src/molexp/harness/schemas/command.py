"""``CommandSpec`` + ``CommandResult`` — executor wire contract (Phase 9 §6.1).

Frozen pydantic models. ``CommandSpec`` describes what to run + how to
constrain it; ``CommandResult`` reports the outcome with artifact refs
for stdout/stderr and any output files the executor discovered.

``env`` is tri-state by intent:

- ``None`` (the default) — inherit the executor process's environment.
- ``{}`` — run with an empty environment (no inheritance).
- non-empty dict — run with exactly those variables.

This avoids the surprise where ``env={}`` silently inherited the parent
env because the executor treated an empty mapping as "unset".
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from molexp.harness.schemas.artifact import ArtifactRef

__all__ = ["CommandResult", "CommandSpec"]


class CommandSpec(BaseModel):
    """Description of one command an :class:`Executor` will run."""

    model_config = ConfigDict(frozen=True)

    cmd: list[str]
    cwd: str
    env: dict[str, str] | None = None
    timeout_s: int = 3600
    expected_outputs: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class CommandResult(BaseModel):
    """Outcome of one :class:`CommandSpec` execution."""

    model_config = ConfigDict(frozen=True)

    exit_code: int
    started_at: datetime
    ended_at: datetime
    stdout_artifact: ArtifactRef
    stderr_artifact: ArtifactRef
    output_artifacts: list[ArtifactRef] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
