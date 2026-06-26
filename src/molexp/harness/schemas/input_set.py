"""``InputSet`` — the parameter-space expansion the workflow runs over.

Step 6 of the plan pipeline. From the concrete :class:`ExperimentSpec` (and
the :class:`WorkflowIR` it produced), this describes *which* root inputs are
swept and over *what* values. It is a declarative specification of the
sweep — the actual cell-by-cell expansion is delegated to the workspace's
``ParamSpace`` family (``GridSpace`` / ``UniformSpace`` in
``molexp.workspace.param``); the harness never reinvents that iteration.

``sweep_axes`` whose ``name`` is not a ``WorkflowIR.inputs`` key is a
validation error (see ``validators.input_set``). A single-value axis is a
legal degenerate sweep (one cell).

Frozen pydantic.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from molexp.harness.schemas.parameter import ParameterSource

__all__ = ["InputSet", "SweepAxis", "SweepStrategy"]


SweepStrategy = Literal["grid", "uniform"]
"""How the axes combine into cells: full Cartesian product, or random sampling."""


class SweepAxis(BaseModel):
    """One swept root input: a named dimension and the values it ranges over."""

    model_config = ConfigDict(frozen=True)

    name: str
    values: list[Any]
    source: ParameterSource = "agent_inferred"
    reason: str | None = None


class InputSet(BaseModel):
    """Declarative parameter-space specification for a bound workflow run."""

    model_config = ConfigDict(frozen=True)

    id: str
    experiment_spec_id: str
    title: str
    sweep_axes: list[SweepAxis] = Field(default_factory=list)
    strategy: SweepStrategy = "grid"
    total_runs: int = 1
    random_seed: int | None = None
