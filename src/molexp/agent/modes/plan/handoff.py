"""``PlanRunHandoff`` — the binding contract between PlanMode and RunMode.

PlanMode (the top-level mode in this module) materializes an
experiment workspace and ends with a human-review approval gate. The
gate's success produces a :class:`PlanRunHandoff`: a frozen pydantic
record carrying every path / identifier a future ``RunMode``
implementation needs to load and execute the generated workflow.

The contract is exposed two ways:

- **In-memory** — :attr:`AgentRunResult.mode_state["plan"]["handoff"]`
  holds the ``PlanRunHandoff.model_dump()`` of the handoff. Consumers
  inside the same Python process consume the dict directly.
- **On disk** — ``HumanReview`` writes the handoff into the
  experiment workspace's ``manifest.yaml`` under the ``handoff`` key
  (``yaml.safe_dump`` of ``json.loads(model_dump_json())``). A future
  ``RunMode`` running outside the agent process loads this and
  reconstitutes the handoff via :meth:`PlanRunHandoff.model_validate`.

``RunMode`` itself is **out of scope** for the sub-spec that lands
this module. Only the contract is published; the runtime that loads
it and drives the workflow lands in a separate spec.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from molexp.agent.modes.plan.plan_folder import PlanManifest, ValidationReport

__all__ = ["PlanRunHandoff"]


class PlanRunHandoff(BaseModel):
    """Frozen contract handed from PlanMode to a future RunMode.

    Carries enough information to materialize and execute the
    generated workflow: the IR paths, the importable entrypoint, and
    snapshots of the manifest and validation report at the moment of
    approval.

    Attributes:
        plan_id: Stable identifier for the plan.
        experiment_workspace_path: Root directory of the materialized workspace.
        workflow_yaml_path: Path to ``ir/workflow.yaml`` (relative to
            ``experiment_workspace_path`` or absolute — the producer
            decides; consumers should respect whichever shape was set).
        source_root: Python source root relative to
            ``experiment_workspace_path``. ``RunMode`` adds this path to
            ``sys.path`` before importing the entrypoint.
        task_ir_paths: Paths to per-task IR files
            (``ir/tasks/<task>.yaml``).
        entrypoint_module: Importable module name (e.g.
            ``"experiment.workflow"``).
        entrypoint_symbol: Module-level symbol that holds the compiled
            :class:`molexp.workflow.Workflow` (e.g. ``"WORKFLOW"``).
        manifest_snapshot: Frozen :class:`PlanManifest` at approval time.
        validation_report_snapshot: Frozen :class:`ValidationReport`
            at approval time.
        created_at: UTC timestamp of handoff creation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_id: str
    experiment_workspace_path: Path
    workflow_yaml_path: Path
    source_root: Path = Path("src")
    task_ir_paths: tuple[Path, ...]
    entrypoint_module: str
    entrypoint_symbol: str
    manifest_snapshot: PlanManifest
    validation_report_snapshot: ValidationReport
    created_at: datetime
