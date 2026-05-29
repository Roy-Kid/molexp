"""LIVE PlanMode demo — a natural-language draft → runnable molexp.workflow code.

Drives :class:`molexp.harness.PlanMode` end-to-end against the **real DeepSeek
API** (``deepseek:deepseek-v4-flash``): a short experiment draft flows through
SaveUserPlan → GenerateExperimentReport → ExtractWorkflowIR → ValidateWorkflowIR
→ BindMolcraftsTasks → ValidateBoundWorkflow → GenerateWorkflowSource →
ValidateWorkflowSource → ApprovalGate, and the generated, validated
``molexp.workflow`` source is printed.

Makes REAL paid DeepSeek API calls. Register the key *in code* — molexp reads
it from ``molexp.config``, never from the environment — by pasting it
into the ``API_KEY`` constant below.
Run directly::

    python examples/harness/plan_mode_live.py

With no key registered it prints a clear message and exits 0 (no traceback). All
network / LLM work is under ``__main__``; importing this module does nothing.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

import molexp

MODEL = "deepseek:deepseek-v4-flash"
DRAFT = "Simulate NEMD ionic mobility of an SPC/E water box under an applied electric field"
API_KEY = ""  # ← paste your DeepSeek key here (registered via molexp.config, not env)


def _run() -> int:
    if not API_KEY:
        print("Set API_KEY at the top of this file and re-run.")
        print("molexp reads the DeepSeek key from molexp.config, never from the environment.")
        return 0
    molexp.config["deepseek_api_key"] = API_KEY

    # Imports are inside _run so module import stays side-effect-free.
    from molexp.agent._pydanticai.router import PydanticAIRouter
    from molexp.agent.router import ModelTier
    from molexp.harness import PlanMode, RouterBackedAgentGateway
    from molexp.harness.prompts import prompts_by_agent
    from molexp.harness.prompts.workflow_source import (
        SYSTEM_PROMPT as WORKFLOW_SOURCE_SYSTEM_PROMPT,
    )
    from molexp.harness.schemas import (
        BoundWorkflow,
        ExperimentReport,
        WorkflowIR,
        WorkflowSource,
    )
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.workspace import Workspace

    with tempfile.TemporaryDirectory() as tmp:
        ws = Workspace(Path(tmp) / "lab", name="plan-mode-live")
        ws.materialize()
        run = ws.add_project("demo").add_experiment("nemd").add_run(parameters={})

        # Gateway shares the run's artifact dir with the Mode-built context
        # (FileArtifactStore is disk-backed, so same-root instances share state).
        store = FileArtifactStore(root=run.run_dir / "artifacts")
        router = PydanticAIRouter(models=dict.fromkeys(ModelTier, MODEL))
        gateway = RouterBackedAgentGateway(
            router=router,
            artifact_store=store,
            agent_responses={
                "experiment_report_writer": ExperimentReport,
                "workflow_ir_extractor": WorkflowIR,
                "bound_workflow_binder": BoundWorkflow,
                "workflow_source_writer": WorkflowSource,
            },
            output_kind_by_agent={
                "experiment_report_writer": "experiment_report",
                "workflow_ir_extractor": "workflow_ir",
                "bound_workflow_binder": "bound_workflow",
                "workflow_source_writer": "workflow_source",
            },
            system_prompt_by_agent={
                **prompts_by_agent(),
                "workflow_source_writer": WORKFLOW_SOURCE_SYSTEM_PROMPT,
            },
            model=MODEL,
        )

        print(f"model : {MODEL}")
        print(f"draft : {DRAFT}")
        print("running PlanMode (real DeepSeek)...\n")
        result = asyncio.run(PlanMode().run(run=run, user_input=DRAFT, gateway=gateway))

        print(f"mode      : {result.mode_name}")
        print(f"run_id    : {result.run_id}")
        print("stage artifacts:")
        for ref in result.stage_artifacts:
            print(f"  {ref.kind:<20} {ref.id}")

        source_refs = [r for r in result.stage_artifacts if r.kind == "workflow_source"]
        if source_refs:
            ws_obj = WorkflowSource.model_validate_json(store.get(source_refs[0].id))
            print("\n--- DeepSeek-generated, validated molexp.workflow source ---\n")
            print(ws_obj.source)
        else:
            print("\n(no workflow_source artifact produced)")
    return 0


if __name__ == "__main__":
    sys.exit(_run())
