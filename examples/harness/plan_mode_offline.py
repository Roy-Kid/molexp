"""Offline PlanMode demo — deterministic, no API key.

The harness counterpart to ``plan_mode_live.py``: same :class:`molexp.harness.PlanMode`
pipeline (draft → ExperimentReport → WorkflowIR → BoundWorkflow → generated,
validated ``molexp.workflow`` source), but driven by a ``StubAgentGateway`` with
canned valid outputs instead of a real LLM. Reproducible anywhere — this is the
example to read first, and the shape the test suite exercises.

Run directly::

    python examples/harness/plan_mode_offline.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

from molexp.harness import PlanMode, generate_audit_report
from molexp.harness.gateways.stub import StubAgentGateway
from molexp.harness.schemas import WorkflowSource
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_event_log import SQLiteEventLog
from molexp.harness.store.sqlite_provenance_store import SQLiteProvenanceStore
from molexp.workspace import Workspace

# A WorkflowSource program that compiles to a real Workflow (public API only).
_VALID_SOURCE = """\
from molexp.workflow import TaskContext, WorkflowBuilder


def build_workflow() -> WorkflowBuilder:
    wf = WorkflowBuilder(name="water_nemd")

    @wf.task
    async def build_system(ctx: TaskContext) -> dict:
        return {"structure": "system.pdb"}

    @wf.task(depends_on=["build_system"])
    async def simulate(ctx: TaskContext) -> dict:
        return {"trajectory": "traj.dcd"}

    return wf
"""

_CANNED = {
    "experiment_report_writer": (
        "experiment_report",
        {
            "title": "Water NEMD",
            "objective": "Measure ionic mobility",
            "system_description": "SPC/E water box under an applied field",
            "experimental_design": "Apply field; record current",
        },
    ),
    "workflow_ir_extractor": (
        "workflow_ir",
        {
            "id": "wf-water",
            "name": "water_nemd",
            "objective": "Compute mobility",
            "inputs": {},
            "tasks": [
                {
                    "id": "build",
                    "name": "Pack water",
                    "purpose": "Build SPC/E box",
                    "task_type": "molecule_builder",
                    "inputs": {},
                    "outputs": {"structure": "structure.pdb"},
                }
            ],
            "edges": [],
            "expected_outputs": [],
        },
    ),
    "bound_workflow_binder": (
        "bound_workflow",
        {
            "id": "bw-water",
            "workflow_ir_id": "wf-water",
            "tasks": [
                {
                    "id": "b-build",
                    "ir_task_id": "build",
                    "capability_id": "molpy.builder.water.SPCEBuilder",
                    "package": "molpy",
                    "callable": "molpy.builder.water.SPCEBuilder.run",
                    "parameters": {},
                    "inputs": {},
                    "outputs": {"structure": "structure.pdb"},
                }
            ],
            "edges": [],
            "execution_backend": "local",
            "environment": {},
            "resource_policy": {
                "backend": "local",
                "max_runtime_s": 3600,
                "denied_paths": ["/", "~/.ssh"],
            },
        },
    ),
    "workflow_source_writer": (
        "workflow_source",
        {
            "source": _VALID_SOURCE,
            "module_name": "water_nemd",
            "bound_workflow_id": "bw-water",
            "symbols": ["WorkflowBuilder", "TaskContext"],
        },
    ),
}


async def _drive(run) -> None:
    store = FileArtifactStore(root=run.run_dir / "artifacts")
    gateway = StubAgentGateway(store)
    for agent_name, (kind, output) in _CANNED.items():
        gateway.register(agent_name, output, output_kind=kind)

    result = await PlanMode().run(run=run, user_input="Simulate water NEMD", gateway=gateway)

    print(f"mode      : {result.mode_name}")
    print(f"run_id    : {result.run_id}")
    print("stage artifacts:")
    for ref in result.stage_artifacts:
        print(f"  {ref.kind:<20} {ref.id}")

    src_ref = next(r for r in result.stage_artifacts if r.kind == "workflow_source")
    ws_obj = WorkflowSource.model_validate_json(store.get(src_ref.id))
    print("\n--- generated, validated molexp.workflow source ---\n")
    print(ws_obj.source)

    # Provenance: the audit report traces the generated code back to the draft.
    provenance = SQLiteProvenanceStore(path=run.run_dir / "harness.sqlite", artifact_store=store)
    lineage = [r.kind for r in provenance.trace_backward(src_ref.id)]
    print(f"workflow_source lineage (ancestors): {lineage}")
    report = generate_audit_report(
        run_id=run.id,
        event_log=SQLiteEventLog(path=run.run_dir / "harness.sqlite"),
        artifact_store=store,
        provenance_store=provenance,
    )
    print(f"audit: {report.summary}")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        ws = Workspace(Path(tmp) / "lab", name="plan-offline")
        ws.materialize()
        run = ws.add_project("demo").add_experiment("nemd").add_run(parameters={})
        asyncio.run(_drive(run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
