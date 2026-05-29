"""End-to-end harness pipeline driven by ``RouterBackedAgentGateway``.

Demonstrates the full 8-stage flow:

::

    SaveUserPlan -> GenerateExperimentReport -> ExtractWorkflowIR
    -> ValidateWorkflowIR -> BindMolcraftsTasks -> ValidateBoundWorkflow
    -> GenerateTestSpec -> ApprovalGate

Each LLM-driven stage routes through :class:`RouterBackedAgentGateway`
backed by a small offline ``_CannedRouter`` that returns pre-built JSON
for each registered ``agent_name``. The router is a minimal
:class:`molexp.agent.router.Router` Protocol impl that mirrors the
``_StubRouter`` shape used in ``tests/test_harness/test_router_backed_gateway.py``.

`pydantic_ai.models.test.TestModel` is imported as the canonical offline
LLM placeholder — a production deployment would wire it into a real
``PydanticAIRouter`` (see ``examples/agent/chat_mode.py``) instead of
this script's hand-rolled ``_CannedRouter``. Either way, no real LLM
provider credentials are required to run this example.

Run directly::

    python examples/harness/full_pipeline.py

Exits 0 and prints the audit-report summary on success. No network access
or API keys needed.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_ai.models.test import (
    TestModel,  # noqa: F401 — imported per spec ac-008; offline-only canonical placeholder
)

from molexp.agent.router import (
    AgenticChunk,
    ModelTier,
    RouterTextResult,
)
from molexp.agent.types import UsageBreakdown
from molexp.harness import (
    ApprovalGate,
    BindMolcraftsTasks,
    ExtractWorkflowIR,
    FileArtifactStore,
    GenerateExperimentReport,
    GenerateTestSpec,
    HarnessRunContext,
    RouterBackedAgentGateway,
    SaveUserPlan,
    SQLiteEventLog,
    SQLiteProvenanceStore,
    StageRunner,
    ValidateBoundWorkflow,
    ValidateWorkflowIR,
    generate_audit_report,
)
from molexp.harness.schemas import (
    ApprovalDecision,
    ApprovalRequest,
    BoundWorkflow,
    ExperimentReport,
    TestSpec,
    WorkflowIR,
)

# ── Per-stage canned responses (deterministic, offline) ──────────────────


def _experiment_report() -> dict[str, Any]:
    return {
        "title": "Water NEMD demo",
        "objective": "Measure ionic mobility",
        "system_description": "SPC/E water box with applied field",
        "experimental_design": "Apply 0.05 V/nm field; record current",
    }


def _workflow_ir() -> dict[str, Any]:
    return {
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
    }


def _bound_workflow() -> dict[str, Any]:
    return {
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
    }


def _test_spec() -> dict[str, Any]:
    return {
        "id": "ts-001",
        "name": "Dry-run sanity",
        "kind": "dry_run_test",
        "description": "Verify build task produces structure.pdb",
        "target_task_id": "build",
        "expected_artifacts": ["structure.pdb"],
    }


# ── Offline router: returns canned JSON in call order ────────────────────


class _CannedRouter:
    """Returns pre-loaded JSON strings in the order they are pushed.

    Implements the structural :class:`molexp.agent.router.Router` Protocol
    without importing any pydantic-ai SDK. Suitable only for deterministic
    offline demos; production setups inject ``molexp.agent._pydanticai.router.PydanticAIRouter``
    or an equivalent wrapper around ``TestModel``.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._index = 0

    async def complete_text(
        self,
        *,
        prompt: str,
        system: str = "",
        message_history: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
    ) -> RouterTextResult:
        if self._index >= len(self._responses):
            raise RuntimeError(f"_CannedRouter exhausted (asked for call #{self._index + 1})")
        text = self._responses[self._index]
        self._index += 1
        return RouterTextResult(text=text, raw=None)

    async def complete_structured(
        self,
        *,
        tier: ModelTier,
        system: str,
        user: str,
        schema: type[BaseModel],
        node_id: str = "",
    ) -> BaseModel:
        text = self._responses[self._index]
        self._index += 1
        return schema.model_validate_json(text)

    def stream_agentic(
        self,
        *,
        prompt: str,
        system: str = "",
        tools: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
        message_history: tuple[Any, ...] = (),
    ) -> AsyncIterator[AgenticChunk]:
        async def _empty() -> AsyncIterator[AgenticChunk]:
            if False:  # pragma: no cover — stream is unused in this demo
                yield  # type: ignore[unreachable]

        return _empty()

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


# ── Driver ───────────────────────────────────────────────────────────────


async def _drive(workspace: Path) -> dict[str, Any]:
    """Execute all 8 stages and return the audit-report payload."""
    artifact_store = FileArtifactStore(root=workspace / "artifacts")
    db_path = workspace / "events.sqlite"
    event_log = SQLiteEventLog(path=db_path)
    provenance = SQLiteProvenanceStore(path=db_path, artifact_store=artifact_store)

    # Router returns the 4 LLM-stage outputs in the order they are called.
    router = _CannedRouter(
        responses=[
            json.dumps(_experiment_report()),
            json.dumps(_workflow_ir()),
            json.dumps(_bound_workflow()),
            json.dumps(_test_spec()),
        ]
    )
    gateway = RouterBackedAgentGateway(
        router=router,
        artifact_store=artifact_store,
        agent_responses={
            "experiment_report_writer": ExperimentReport,
            "workflow_ir_extractor": WorkflowIR,
            "bound_workflow_binder": BoundWorkflow,
            "test_spec_writer": TestSpec,
        },
        output_kind_by_agent={
            "experiment_report_writer": "experiment_report",
            "workflow_ir_extractor": "workflow_ir",
            "bound_workflow_binder": "bound_workflow",
            "test_spec_writer": "test_spec",
        },
    )

    ctx = HarnessRunContext(
        run_id="run-full-pipeline",
        workspace_root=workspace,
        artifact_store=artifact_store,
        event_log=event_log,
        provenance_store=provenance,
        agent_gateway=gateway,
    )
    runner = StageRunner(ctx)

    user_plan = await runner.run_stage(SaveUserPlan(user_text="Simulate water NEMD"))
    report = await runner.run_stage(GenerateExperimentReport())
    workflow_ir = await runner.run_stage(ExtractWorkflowIR())
    ir_validation = await runner.run_stage(ValidateWorkflowIR())
    bound_wf = await runner.run_stage(BindMolcraftsTasks())
    bw_validation = await runner.run_stage(ValidateBoundWorkflow())
    test_spec = await runner.run_stage(GenerateTestSpec())
    # ApprovalGate takes pre-resolved (request, decision) pairs. In a real
    # interactive pipeline the orchestrator would surface the request to a
    # human reviewer; here we auto-grant for the offline demo.
    now = datetime.now(tz=UTC)
    approval_request = ApprovalRequest(
        id="approval-final-report",
        intent="final_report",
        reason="auto-approve final pipeline output for offline demo",
        triggered_by_policy="examples/harness/full_pipeline.py",
        created_at=now,
    )
    approval_decision = ApprovalDecision(
        request_id=approval_request.id,
        granted=True,
        decided_by="full_pipeline_example",
        decided_at=now,
        reason="offline auto-approval",
    )
    approval = await runner.run_stage(
        ApprovalGate(
            decisions=[(approval_request, approval_decision)],
            subject_artifact_ids=[test_spec.id, bound_wf.id],
        )
    )

    return {
        "artifacts": {
            "user_plan": user_plan.id,
            "experiment_report": report.id,
            "workflow_ir": workflow_ir.id,
            "ir_validation": ir_validation.id,
            "bound_workflow": bound_wf.id,
            "bw_validation": bw_validation.id,
            "test_spec": test_spec.id,
            "approval": approval.id,
        },
        "audit_report": generate_audit_report(
            run_id=ctx.run_id,
            event_log=ctx.event_log,
            artifact_store=ctx.artifact_store,
            provenance_store=ctx.provenance_store,
        ).model_dump(mode="json"),
    }


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        result = asyncio.run(_drive(workspace))
        print(f"workspace : {workspace}")
        print(f"artifacts : {len(result['artifacts'])} stage outputs")
        for name, art_id in result["artifacts"].items():
            print(f"  {name:<20} {art_id}")
        print()
        print("audit_report:")
        print(json.dumps(result["audit_report"], indent=2)[:600] + " ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
