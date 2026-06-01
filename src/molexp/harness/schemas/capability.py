"""``ToolCapability`` — typed catalog entry for one harness-invokable tool.

Per ``.claude/notes/harness-goal.md`` §5.2: every Molcrafts capability that
the harness can dispatch (a Python callable, a CLI invocation, an MCP tool,
a Slurm job template) is described by one frozen ``ToolCapability``. The
:class:`CapabilityRegistry` stores them keyed by ``id``; the
:func:`validate_bound_workflow` extension cross-checks ``BoundTask`` fields
against the registered capability so an agent cannot bind a workflow to a
tool that doesn't exist.

Phase 4 stores ``input_schema`` / ``output_schema`` as free-form ``dict``
(no jsonschema dep); shallow validation lives in
:meth:`InMemoryCapabilityRegistry.validate_call`. Deep value-type checking
is Phase 5+ if needed.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ToolCapability"]


class ToolCapability(BaseModel):
    """One harness-invokable capability."""

    model_config = ConfigDict(frozen=True)

    id: str
    package: str
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    callable_path: str | None = None
    cli_template: list[str] | None = None
    side_effects: list[str] = Field(default_factory=list)
    supported_backends: list[str] = Field(default_factory=lambda: ["local"])
    examples: list[dict[str, Any]] = Field(default_factory=list)
    version: str | None = None
    tags: list[str] = Field(default_factory=list)
