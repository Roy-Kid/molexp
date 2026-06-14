"""NL experiment goal → plan → generated tests → REAL execution → final report.

The flagship harness demo: one natural-language goal flows through both
harness modes on a single ``workspace.Run``:

- ``PlanMode`` (9 stages) — ``SaveUserPlan → GenerateExperimentReport →
  ExtractWorkflowIR → ValidateWorkflowIR → BindMolcraftsTasks →
  ValidateBoundWorkflow → GenerateWorkflowSource → ValidateWorkflowSource →
  ApprovalGate`` — expands the goal into an experiment plan and validated,
  runnable ``molexp.workflow`` source.
- ``RunMode`` (10 stages) — ``GenerateTestSpec → ValidateTestSpec →
  GenerateTestCode → ValidateTestSource → MaterializeExecution →
  ExecuteTests → ExecuteWorkflow → GenerateFinalReport → ApprovalGate →
  GenerateAuditReport`` — writes unit tests for the generated workflow,
  REALLY runs them with pytest, executes the workflow on the real
  ``molexp.workflow`` engine in an executor subprocess, and writes the final
  experiment report from the actual results.

OFFLINE BY DEFAULT — zero network, zero API keys, deterministic: the
in-file :class:`CannedGateway` implements the public ``AgentGateway``
Protocol (the same seam the production ``RouterBackedAgentGateway`` plugs
into) and serves pre-authored responses for all seven LLM agents. Only the
LLM is canned: every validator runs for real, pytest runs for real, and the
workflow really executes — the canned experiment is a 1D random walk whose
diffusion coefficient is estimated via Einstein's relation D = MSD/(2·d·t).

LIVE MODE — paste a DeepSeek key into ``API_KEY`` below (molexp reads LLM
keys from ``molexp.config``, registered in code, never from the
environment) and the same pipeline runs against the real model through
``RouterBackedAgentGateway``.

Run directly::

    python examples/harness/experiment_pipeline.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

import molexp
from molexp.harness import (
    AgentCallResult,
    AgentCallSpec,
    AgentGateway,
    AgentResponseNotRegisteredError,
    ArtifactRef,
    BoundTask,
    BoundWorkflow,
    DependencyEdge,
    ExecutionResult,
    ExpectedOutput,
    ExperimentReport,
    FinalReport,
    ParameterValue,
    PlanMode,
    ResourcePolicy,
    RunMode,
    TaskIR,
    TestSource,
    TestSpec,
    WorkflowIR,
    WorkflowSource,
)
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.workspace import Workspace

MODEL = "deepseek:deepseek-v4-flash"
API_KEY = ""  # ← paste your DeepSeek key here for live mode (in-code key law)

GOAL = (
    "Estimate the diffusion coefficient of a 1D random walker from the "
    "mean squared displacement of an ensemble of seeded walks."
)

# ─────────────────────────────────────────────────────────────────────────────
# The canned "LLM outputs". The workflow + test sources are REAL programs:
# ValidateWorkflowSource compiles the workflow, pytest really runs the tests,
# and the materialized driver really executes the workflow engine.
# ─────────────────────────────────────────────────────────────────────────────

WORKFLOW_SOURCE = '''\
"""Random-walk diffusion workflow (generated for the molexp harness demo)."""

import random

from molexp.workflow import TaskContext, WorkflowCompiler

SEED = 20260610
N_WALKERS = 200
N_STEPS = 400


def random_walk_displacements(n_walkers, n_steps, seed):
    """Final displacements of ``n_walkers`` independent ±1-step walks."""
    rng = random.Random(seed)
    finals = []
    for _ in range(n_walkers):
        x = 0
        for _ in range(n_steps):
            x += 1 if rng.random() < 0.5 else -1
        finals.append(x)
    return finals


def mean_squared_displacement(displacements):
    return sum(d * d for d in displacements) / len(displacements)


def estimate_diffusion_coefficient(msd, total_time, dimensions=1):
    """Einstein relation: D = MSD / (2 * d * t)."""
    return msd / (2 * dimensions * total_time)


def build_workflow() -> WorkflowCompiler:
    wf = WorkflowCompiler(name="random_walk_diffusion")

    @wf.task
    async def generate_walks(ctx: TaskContext) -> dict:
        finals = random_walk_displacements(N_WALKERS, N_STEPS, SEED)
        return {"displacements": finals, "n_steps": N_STEPS}

    @wf.task(depends_on=["generate_walks"])
    async def compute_msd(ctx: TaskContext) -> dict:
        walks = ctx.inputs  # single upstream → its output arrives directly
        msd = mean_squared_displacement(walks["displacements"])
        return {"msd": msd, "n_steps": walks["n_steps"]}

    @wf.task(depends_on=["compute_msd"])
    async def estimate_d(ctx: TaskContext) -> dict:
        upstream = ctx.inputs
        d = estimate_diffusion_coefficient(upstream["msd"], upstream["n_steps"])
        return {"diffusion_coefficient": d, "msd": upstream["msd"]}

    return wf
'''

TEST_SOURCE = '''\
"""Generated unit tests for the random-walk diffusion workflow."""

from generated_workflow import (
    N_STEPS,
    N_WALKERS,
    SEED,
    build_workflow,
    estimate_diffusion_coefficient,
    mean_squared_displacement,
    random_walk_displacements,
)


def test_walks_are_deterministic_with_seed():
    a = random_walk_displacements(N_WALKERS, N_STEPS, SEED)
    b = random_walk_displacements(N_WALKERS, N_STEPS, SEED)
    assert a == b
    assert len(a) == N_WALKERS


def test_msd_is_positive_and_near_n_steps():
    walks = random_walk_displacements(N_WALKERS, N_STEPS, SEED)
    msd = mean_squared_displacement(walks)
    assert msd > 0
    # Var of an n-step ±1 walk is exactly n; generous statistical bounds.
    assert 0.5 * N_STEPS < msd < 2.0 * N_STEPS


def test_diffusion_coefficient_is_positive_and_near_half():
    walks = random_walk_displacements(N_WALKERS, N_STEPS, SEED)
    msd = mean_squared_displacement(walks)
    d = estimate_diffusion_coefficient(msd, N_STEPS)
    assert d > 0
    assert 0.25 < d < 1.0  # unit-time ±1 walk → D ≈ 0.5


def test_workflow_compiles_with_three_tasks():
    assert build_workflow().compile() is not None
'''


def _canned_responses() -> dict[str, tuple[str, dict]]:
    """Build {agent_name: (output_kind, payload)} from REAL schema instances.

    Constructing through the schemas (then ``model_dump``) means any schema
    drift in molexp.harness turns this example red in the smoke gate —
    exactly the anti-drift job the example tier exists for.

    The numeric literals below (200 walkers / 400 steps / seed 20260610)
    must stay in sync with the SEED / N_WALKERS / N_STEPS constants inside
    WORKFLOW_SOURCE — they describe the same "generated" program.
    """
    report = ExperimentReport(
        title="Random-walk diffusion",
        objective=GOAL,
        system_description="An ensemble of 200 independent 1D ±1-step random walkers.",
        experimental_design=(
            "Generate seeded walks, compute the ensemble MSD at t = 400 steps, "
            "and estimate D via Einstein's relation D = MSD/(2 d t)."
        ),
        expected_outputs=["diffusion_coefficient"],
    )
    ir = WorkflowIR(
        id="wf-random-walk",
        name="random_walk_diffusion",
        objective="Estimate D from the MSD of seeded 1D random walks",
        inputs={
            "n_walkers": ParameterValue(value=200, source="agent_inferred", approved=True),
            "n_steps": ParameterValue(value=400, source="agent_inferred", approved=True),
            "seed": ParameterValue(value=20260610, source="agent_inferred", approved=True),
        },
        tasks=[
            TaskIR(
                id="generate_walks",
                name="Generate walks",
                purpose="Simulate the seeded walker ensemble",
                task_type="simulation",
                inputs={},
                outputs={"displacements": "final displacements"},
            ),
            TaskIR(
                id="compute_msd",
                name="Compute MSD",
                purpose="Ensemble mean squared displacement",
                task_type="analysis",
                inputs={},
                outputs={"msd": "mean squared displacement"},
            ),
            TaskIR(
                id="estimate_d",
                name="Estimate D",
                purpose="Einstein relation D = MSD/(2 d t)",
                task_type="analysis",
                inputs={},
                outputs={"diffusion_coefficient": "estimated D"},
            ),
        ],
        edges=[
            DependencyEdge(source_task_id="generate_walks", target_task_id="compute_msd"),
            DependencyEdge(source_task_id="compute_msd", target_task_id="estimate_d"),
        ],
        expected_outputs=[
            ExpectedOutput(
                name="diffusion_coefficient",
                kind="analysis_result",
                description="Estimated D from the Einstein relation",
            )
        ],
    )
    bound = BoundWorkflow(
        id="bw-random-walk",
        workflow_ir_id=ir.id,
        tasks=[
            BoundTask(
                id=f"b-{task.id}",
                ir_task_id=task.id,
                capability_id=f"stdlib.random_walk.{task.id}",
                package="python-stdlib",
                callable=f"generated_workflow.{task.id}",
                parameters={},
                inputs={},
                outputs=dict.fromkeys(task.outputs, "json"),
            )
            for task in ir.tasks
        ],
        edges=[
            DependencyEdge(source_task_id="b-generate_walks", target_task_id="b-compute_msd"),
            DependencyEdge(source_task_id="b-compute_msd", target_task_id="b-estimate_d"),
        ],
        execution_backend="local",
        environment={},
        resource_policy=ResourcePolicy(
            backend="local", max_runtime_s=600, denied_paths=["/", "~/.ssh"]
        ),
    )
    workflow_source = WorkflowSource(
        source=WORKFLOW_SOURCE,
        module_name="generated_workflow",
        bound_workflow_id=bound.id,
        symbols=("WorkflowCompiler", "TaskContext"),
    )
    test_spec = TestSpec(
        id="ts-random-walk",
        name="random-walk unit tests",
        kind="unit_test",
        target_task_id="estimate_d",
        description="Determinism, MSD sanity, and D = MSD/(2 d t) bounds.",
    )
    test_source = TestSource(
        source=TEST_SOURCE,
        module_name="test_generated_workflow",
        test_spec_id=test_spec.id,
        bound_workflow_id=bound.id,
        symbols=("build_workflow",),
    )
    final_report = FinalReport(
        title="Random-walk diffusion — final report",
        objective=GOAL,
        methods_summary="Three-task molexp.workflow executed via the harness driver.",
        test_summary="4 generated pytest cases passed (determinism, MSD, D bounds).",
        execution_summary="Workflow executed to completion in an executor subprocess.",
        results="Estimated D ≈ 0.5 in lattice units (MSD/(2t) of the seeded ensemble).",
        conclusions="The seeded ensemble reproduces Einstein diffusion: D = MSD/(2 d t).",
        limitations=["toy lattice walk", "single ensemble size"],
        next_steps=["sweep n_steps", "compare against an analytic ±1-walk variance"],
    )
    return {
        "experiment_report_writer": ("experiment_report", report.model_dump(mode="json")),
        "workflow_ir_extractor": ("workflow_ir", ir.model_dump(mode="json")),
        "bound_workflow_binder": ("bound_workflow", bound.model_dump(mode="json")),
        "workflow_source_writer": ("workflow_source", workflow_source.model_dump(mode="json")),
        "test_spec_writer": ("test_spec", test_spec.model_dump(mode="json")),
        "test_code_writer": ("test_source", test_source.model_dump(mode="json")),
        "final_report_writer": ("final_report", final_report.model_dump(mode="json")),
    }


class CannedGateway:
    """In-file ``AgentGateway`` — the Protocol seam, with canned responses.

    Anyone can stand a backend behind ``molexp.harness.AgentGateway``: the
    only contract is ``async call(spec) -> AgentCallResult`` plus the
    persistence law the shipped gateways follow — persist the RAW response
    first (kind ``log``), then the PARSED output (the per-agent kind), both
    ``created_by="agent:<name>"`` with ``parent_ids`` mirroring the call's
    ``input_artifact_ids``, so lineage stays intact offline.
    """

    def __init__(self, store: FileArtifactStore, responses: dict[str, tuple[str, dict]]) -> None:
        self._store = store
        self._responses = responses
        self.calls: list[tuple[AgentCallSpec, AgentCallResult]] = []

    async def call(self, spec: AgentCallSpec) -> AgentCallResult:
        if spec.agent_name not in self._responses:
            raise AgentResponseNotRegisteredError(
                f"no canned response registered for agent {spec.agent_name!r}"
            )
        kind, payload = self._responses[spec.agent_name]
        created_by = f"agent:{spec.agent_name}"
        raw_ref = self._store.put_text(
            kind="log",
            text=json.dumps(payload, sort_keys=True),
            created_by=created_by,
            parent_ids=list(spec.input_artifact_ids),
        )
        out_ref = self._store.put_json(
            kind=kind,
            obj=payload,
            created_by=created_by,
            parent_ids=list(spec.input_artifact_ids),
        )
        result = AgentCallResult(
            output_artifact=out_ref,
            raw_response_artifact=raw_ref,
            model="canned",
            usage={},
        )
        self.calls.append((spec, result))
        return result


def _live_gateway(store: FileArtifactStore) -> AgentGateway:
    """The same pipeline against a real LLM (paste API_KEY above)."""
    from molexp.agent import PydanticAIRouter  # public lazy re-export, never _pydanticai
    from molexp.agent.router import ModelTier
    from molexp.harness import RouterBackedAgentGateway
    from molexp.harness.prompts import prompts_by_agent
    from molexp.harness.prompts.workflow_source import (
        SYSTEM_PROMPT as WORKFLOW_SOURCE_SYSTEM_PROMPT,
    )

    molexp.config["deepseek_api_key"] = API_KEY
    schemas = {
        "experiment_report_writer": ExperimentReport,
        "workflow_ir_extractor": WorkflowIR,
        "bound_workflow_binder": BoundWorkflow,
        "workflow_source_writer": WorkflowSource,
        "test_spec_writer": TestSpec,
        "test_code_writer": TestSource,
        "final_report_writer": FinalReport,
    }
    kinds = {name: kind for name, (kind, _payload) in _canned_responses().items()}
    return RouterBackedAgentGateway(
        router=PydanticAIRouter(models=dict.fromkeys(ModelTier, MODEL)),
        artifact_store=store,
        agent_responses=schemas,
        output_kind_by_agent=kinds,
        system_prompt_by_agent={
            **prompts_by_agent(),
            "workflow_source_writer": WORKFLOW_SOURCE_SYSTEM_PROMPT,
        },
        model=MODEL,
    )


def _print_stage_table(label: str, names: list[str], refs: tuple[ArtifactRef, ...]) -> None:
    print(f"\n{label}")
    for name, ref in zip(names, refs, strict=True):
        print(f"  {name:<26} {ref.kind:<20} {ref.id}")


def _assert_gateway_contract(gateway: CannedGateway) -> None:
    """Self-check: the in-file gateway really honors the persistence law."""
    assert isinstance(gateway, AgentGateway), "CannedGateway must satisfy the Protocol"
    spec, result = gateway.calls[0]
    assert result.raw_response_artifact is not None
    assert result.raw_response_artifact.kind == "log"
    assert result.output_artifact.parent_ids == list(spec.input_artifact_ids)
    assert result.raw_response_artifact.parent_ids == list(spec.input_artifact_ids)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        ws = Workspace(Path(tmp) / "lab", name="experiment-pipeline")
        ws.materialize()
        run = ws.add_project("demo").add_experiment("random-walk").add_run(params={})
        store = FileArtifactStore(root=run.run_dir / "artifacts")

        offline = not API_KEY
        gateway: AgentGateway = (
            CannedGateway(store, _canned_responses()) if offline else _live_gateway(store)
        )
        print(f"goal    : {GOAL}")
        print(f"mode    : {'offline (canned LLM, real engine)' if offline else f'live ({MODEL})'}")
        print(f"run     : {run.id}")

        plan = PlanMode()
        plan_names = [s.name for s in plan.stages(GOAL)]
        plan_result = asyncio.run(plan.run(run=run, user_input=GOAL, gateway=gateway))
        _print_stage_table("PlanMode stage artifacts:", plan_names, plan_result.stage_artifacts)
        if isinstance(gateway, CannedGateway):
            _assert_gateway_contract(gateway)

        run_mode = RunMode()  # default LocalExecutor → real pytest + real engine
        run_names = [s.name for s in run_mode.stages(GOAL)]
        run_result = asyncio.run(run_mode.run(run=run, user_input=GOAL, gateway=gateway))
        _print_stage_table("RunMode stage artifacts:", run_names, run_result.stage_artifacts)

        # The generated tests really ran — show pytest's own summary.
        test_ref = next(a for a in run_result.stage_artifacts if a.kind == "test_result")
        test_payload = json.loads(store.get(test_ref.id))
        pytest_stdout = store.get(test_payload["stdout"]["id"]).decode("utf-8")
        print(f"\npytest  : {pytest_stdout.strip().splitlines()[-1]}")

        # The workflow really executed — pull D out of the engine's outputs.
        exec_ref = next(a for a in run_result.stage_artifacts if a.kind == "execution_result")
        execution = ExecutionResult.model_validate_json(store.get(exec_ref.id))
        assert execution.status == "succeeded"
        d_value = float(execution.outputs["estimate_d"]["diffusion_coefficient"])
        assert d_value > 0, "Einstein relation must give a positive D"
        print(f"engine  : status={execution.status} exit={execution.exit_code}")
        print(f"D       : {d_value:.4f} (lattice units; expectation ≈ 0.5)")

        # The final report is a real artifact written from those results.
        report_ref = next(a for a in run_result.stage_artifacts if a.kind == "final_report")
        report = FinalReport.model_validate_json(store.get(report_ref.id))
        print(f"\nFinal report — {report.title}")
        print(f"  objective   : {report.objective}")
        print(f"  tests       : {report.test_summary}")
        print(f"  execution   : {report.execution_summary}")
        print(f"  results     : {report.results}")
        print(f"  conclusions : {report.conclusions}")
        print(f"  next steps  : {'; '.join(report.next_steps)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
