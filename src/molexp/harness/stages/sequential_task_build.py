"""``SequentialTaskBuild`` — parallel task codegen, serial pytest.

E2E lesson: whole-suite codegen fails late; per-task pytest catches API
mistakes early. molmcp covers **domain** packages (molpy/molpack), **not**
molexp workflow wiring (return-dict → next-task params) — that contract is
injected from the bound workflow.

Flow:

1. **Parallel** HEAVY codegen for every task module (shared wiring contract).
2. **Parallel** HEAVY unit-test modules for every task (given its module).
3. **Serial** (topo order): materialize + ``pytest tests/test_<slug>.py``;
   on red, repair that task only (≤ attempts) then re-pytest.
4. Assemble full package + full-suite pytest.

Downstream :class:`CompileWorkflow` still proves the assembled package compiles.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import TYPE_CHECKING, ClassVar

import yaml
from mollog import get_logger

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError
from molexp.harness.prompts.capability_catalog import render_capability_catalog
from molexp.harness.prompts.codegen_prompt import wiring_dict_from_bound
from molexp.harness.schemas import (
    BoundWorkflow,
    GeneratedFile,
    PlanArtifactRef,
    TestSource,
    WorkflowSource,
)
from molexp.harness.stages._resolve import require_latest
from molexp.harness.stages.evidence import collect_evidence_text, diagnose_failure
from molexp.harness.stages.generate_workflow_source import (
    _rename_top_function,
    _render_assembly,
    _slug,
)
from molexp.harness.stages.materialize_execution import MaterializeExecution
from molexp.harness.stages.task_codegen import (
    compose_task_markdown,
    failure_text,
    gen_task_module,
    gen_task_test,
    run_pytest,
)

if TYPE_CHECKING:
    from molexp.harness.executors import Executor

__all__ = ["SequentialTaskBuild"]

_LOG = get_logger(__name__)


def _is_dry_run_executor(executor: object) -> bool:
    return type(executor).__name__ == "DryRunExecutor"


class SequentialTaskBuild(Stage):
    """Parallel task codegen; serial pytest with per-task repair."""

    name: ClassVar[str] = "sequential_task_build"

    def __init__(self, executor: Executor, *, attempts: int = 3, timeout_s: int = 300) -> None:
        if attempts < 1:
            raise ValueError("SequentialTaskBuild attempts must be >= 1")
        self._executor = executor
        self._attempts = attempts
        self._timeout_s = timeout_s
        self._materialize = MaterializeExecution()
        self._force_skip_pytest = _is_dry_run_executor(executor)
        # Per-stage cache: task_id → task_context_pack artifact id (parallel-safe).
        self._context_pack_ids: dict[str, str] = {}

    async def run(self, ctx: HarnessRunContext) -> PlanArtifactRef:
        if ctx.agent_gateway is None:
            raise StageExecutionError("SequentialTaskBuild requires ctx.agent_gateway")
        bound_ref = require_latest(ctx, "bound_workflow", stage=self.name)
        experiment_spec = require_latest(ctx, "experiment_spec", stage=self.name)
        bound = BoundWorkflow.model_validate_json(ctx.artifact_store.get(bound_ref.id))
        if not bound.tasks:
            raise StageExecutionError("SequentialTaskBuild: bound_workflow has no tasks")

        catalog_id = self._catalog_prompt(ctx, bound_ref.id)
        slugs = {t.id: _slug(t.ir_task_id or t.id) for t in bound.tasks}
        order = _topo_order(bound)
        stage_t0 = time.monotonic()

        # molexp workflow wiring is NOT in molmcp — inject bound IO contract.
        wiring_id = self._wiring_contract_artifact(ctx, bound=bound, slugs=slugs)

        # ── Phase 1: PARALLEL module codegen for all tasks ─────────────────
        _LOG.info(
            f"[sequential_task_build] parallel codegen n={len(order)} "
            f"(molmcp=domain only; molexp IO contract injected)"
        )
        gen_t0 = time.monotonic()

        async def _code_one(task: object) -> tuple[str, str]:
            slug = slugs[task.id]
            context_id = self._task_context_pack(ctx, task=task, slug=slug)
            src = await self._gen_task_module(
                ctx,
                bound=bound,
                bound_ref=bound_ref,
                task=task,
                experiment_spec_id=experiment_spec.id,
                catalog_id=catalog_id,
                feedback_ids=[],
                context_id=context_id,
                wiring_id=wiring_id,
                is_repair=False,
                slugs=slugs,
            )
            return task.id, _rename_top_function(src, slug)

        code_pairs = await asyncio.gather(*[_code_one(t) for t in order])
        modules: dict[str, str] = dict(code_pairs)
        _LOG.info(
            f"[sequential_task_build] parallel codegen done "
            f"duration_s={time.monotonic() - gen_t0:.2f}"
        )

        # ── Phase 2: PARALLEL unit-test modules (given source) ─────────────
        test_t0 = time.monotonic()

        async def _test_one(task: object) -> tuple[str, str]:
            slug = slugs[task.id]
            context_id = self._task_context_pack(ctx, task=task, slug=slug)
            src = await self._gen_task_test(
                ctx,
                bound_ref=bound_ref,
                task=task,
                slug=slug,
                module_src=modules[task.id],
                feedback_ids=[],
                context_id=context_id,
                wiring_id=wiring_id,
                is_repair=False,
                bound=bound,
                slugs=slugs,
            )
            return task.id, src

        test_pairs = await asyncio.gather(*[_test_one(t) for t in order])
        tests: dict[str, str] = dict(test_pairs)
        _LOG.info(
            f"[sequential_task_build] parallel test codegen done "
            f"duration_s={time.monotonic() - test_t0:.2f}"
        )

        wf_files: dict[str, GeneratedFile] = {
            f"workflow/{slugs[tid]}.py": GeneratedFile(path=f"workflow/{slugs[tid]}.py", source=src)
            for tid, src in modules.items()
        }
        test_files: dict[str, GeneratedFile] = {
            f"tests/test_{slugs[tid]}.py": GeneratedFile(
                path=f"tests/test_{slugs[tid]}.py", source=src
            )
            for tid, src in tests.items()
        }

        # ── Phase 3: SERIAL pytest in topo order + per-task repair ─────────
        last_result: PlanArtifactRef | None = None
        for task in order:
            slug = slugs[task.id]
            task_t0 = time.monotonic()
            test_path = f"tests/test_{slug}.py"
            _LOG.info(
                f"[sequential_task_build] serial pytest task={task.id!r} "
                f"slug={slug} budget={self._attempts}"
            )
            module_src, test_src = await self._repair_until_green(
                ctx,
                bound=bound,
                bound_ref=bound_ref,
                task=task,
                slug=slug,
                experiment_spec_id=experiment_spec.id,
                catalog_id=catalog_id,
                wiring_id=wiring_id,
                module_src=modules[task.id],
                test_src=tests[task.id],
                wf_files=wf_files,
                test_files=test_files,
                slugs=slugs,
            )
            modules[task.id] = module_src
            tests[task.id] = test_src
            wf_files[f"workflow/{slug}.py"] = GeneratedFile(
                path=f"workflow/{slug}.py", source=module_src
            )
            test_files[test_path] = GeneratedFile(path=test_path, source=test_src)
            self._publish_cumulative(
                ctx,
                bound=bound,
                bound_ref=bound_ref,
                slugs=slugs,
                wf_files=wf_files,
                test_files=test_files,
            )
            last_result = await self._pytest_one(ctx, test_path=test_path, task_id=task.id)
            _LOG.info(
                f"[sequential_task_build] task {task.id!r} green "
                f"duration_s={time.monotonic() - task_t0:.2f} "
                f"elapsed_s={time.monotonic() - stage_t0:.2f}"
            )

        assert last_result is not None
        result = await self._pytest_all(ctx, test_files=test_files)
        _LOG.info(
            f"[sequential_task_build] all tasks green "
            f"n={len(order)} total_s={time.monotonic() - stage_t0:.2f}"
        )
        return result

    async def _repair_until_green(
        self,
        ctx: HarnessRunContext,
        *,
        bound: BoundWorkflow,
        bound_ref: PlanArtifactRef,
        task: object,
        slug: str,
        experiment_spec_id: str,
        catalog_id: str | None,
        wiring_id: str | None,
        module_src: str,
        test_src: str,
        wf_files: dict[str, GeneratedFile],
        test_files: dict[str, GeneratedFile],
        slugs: dict[str, str],
    ) -> tuple[str, str]:
        """Serial pytest + repair loop for one task (impl/test rewrite)."""
        last_err: StageExecutionError | None = None
        for attempt in range(1, self._attempts + 1):
            attempt_t0 = time.monotonic()
            trial_wf = {
                **wf_files,
                f"workflow/{slug}.py": GeneratedFile(path=f"workflow/{slug}.py", source=module_src),
            }
            trial_tests = {
                **test_files,
                f"tests/test_{slug}.py": GeneratedFile(
                    path=f"tests/test_{slug}.py", source=test_src
                ),
            }
            self._publish_cumulative(
                ctx,
                bound=bound,
                bound_ref=bound_ref,
                slugs=slugs,
                wf_files=trial_wf,
                test_files=trial_tests,
            )
            try:
                pytest_t0 = time.monotonic()
                await self._pytest_one(ctx, test_path=f"tests/test_{slug}.py", task_id=task.id)
                _LOG.info(
                    f"[sequential_task_build] task {task.id!r} attempt "
                    f"{attempt}/{self._attempts} ok "
                    f"pytest_s={time.monotonic() - pytest_t0:.2f} "
                    f"attempt_s={time.monotonic() - attempt_t0:.2f}"
                )
                return module_src, test_src
            except StageExecutionError as exc:
                last_err = exc
                if attempt >= self._attempts:
                    break
                text = self._failure_text(ctx, exc)
                diagnosis = diagnose_failure(exc, text)
                evidence = collect_evidence_text(diagnosis, ctx=ctx)
                kind = _classify_failure(text)
                body = (
                    f"## task {task.id} attempt {attempt}/{self._attempts} "
                    f"failed ({kind})\n\n{text}\n\n{evidence}\n\n"
                    "Rewrite the GUILTY artifact only: "
                    f"{'workflow task module' if kind == 'impl' else 'pytest module'}. "
                    "molmcp is for domain APIs only; molexp dataflow is "
                    "return-dict keys → next-task parameter names (see wiring "
                    "contract). Prefer live source signatures over inventing APIs."
                )
                fb = ctx.artifact_store.put_text(
                    kind="task_build_feedback",
                    text=body,
                    created_by=self.name,
                    parent_ids=[],
                )
                context_id = self._task_context_pack(ctx, task=task, slug=slug)
                feedback_ids = [fb.id]
                gen_t0 = time.monotonic()
                if kind == "impl":
                    module_src = await self._gen_task_module(
                        ctx,
                        bound=bound,
                        bound_ref=bound_ref,
                        task=task,
                        experiment_spec_id=experiment_spec_id,
                        catalog_id=catalog_id,
                        feedback_ids=feedback_ids,
                        context_id=context_id,
                        wiring_id=wiring_id,
                        is_repair=True,
                        slugs=slugs,
                    )
                    module_src = _rename_top_function(module_src, slug)
                else:
                    test_src = await self._gen_task_test(
                        ctx,
                        bound_ref=bound_ref,
                        task=task,
                        slug=slug,
                        module_src=module_src,
                        feedback_ids=feedback_ids,
                        context_id=context_id,
                        wiring_id=wiring_id,
                        is_repair=True,
                        bound=bound,
                        slugs=slugs,
                    )
                _LOG.warning(
                    f"[sequential_task_build] task {task.id!r} attempt "
                    f"{attempt}/{self._attempts} red ({kind}) "
                    f"repair_gen_s={time.monotonic() - gen_t0:.2f} — rewriting"
                )
        assert last_err is not None
        raise last_err

    def _wiring_contract_artifact(
        self,
        ctx: HarnessRunContext,
        *,
        bound: BoundWorkflow,
        slugs: dict[str, str],
    ) -> str:
        """Persist inter-task IO contract as YAML (same shape as codegen wiring).

        molmcp covers domain libs only; this map is molexp-owned. Authors also
        receive the same structure inline via :func:`assemble_task_codegen_prompt`.
        """
        doc = {
            "kind": "molexp.workflow_wiring_contract",
            "note": (
                "molmcp documents domain libraries (molpy/molpack). "
                "It does not document how molexp wires tasks."
            ),
            "wiring": wiring_dict_from_bound(bound, slugs=slugs),
        }
        ref = ctx.artifact_store.put_text(
            kind="workflow_wiring_contract",
            text=yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, default_flow_style=False),
            created_by=f"stage:{self.name}",
            parent_ids=[],
        )
        return ref.id

    def _task_context_pack(self, ctx: HarnessRunContext, *, task: object, slug: str) -> str | None:
        """Compose molmcp knowledge pages once per task (no agent tool loop).

        Injected into first-pass YAML prompts as ``domain_context`` so HEAVY
        codegen does not need a live MCP session. Cached in-memory on this
        stage instance (multi-task parallel-safe) and on the artifact store.
        """
        tid = str(getattr(task, "id", slug))
        cached = self._context_pack_ids.get(tid)
        if cached is not None:
            return cached
        cache_key = f"{tid}:{slug}"
        task_text = (
            f"{getattr(task, 'ir_task_id', None) or tid} {slug} "
            f"{getattr(task, 'description', '') or ''} "
            f"{getattr(task, 'capability_id', '') or ''}"
        )
        md = compose_task_markdown(task_text)
        if not md:
            return None
        ref = ctx.artifact_store.put_text(
            kind="task_context_pack",
            text=f"# task_context {cache_key}\n\n{md}",
            created_by=f"stage:{self.name}",
            parent_ids=[],
        )
        self._context_pack_ids[tid] = ref.id
        return ref.id

    async def _gen_task_module(
        self,
        ctx: HarnessRunContext,
        *,
        bound: BoundWorkflow,
        bound_ref: PlanArtifactRef,
        task: object,
        experiment_spec_id: str,
        catalog_id: str | None,
        feedback_ids: list[str],
        context_id: str | None = None,
        wiring_id: str | None = None,  # noqa: ARG002 — kept for call-site compat
        is_repair: bool = False,
        slugs: dict[str, str] | None = None,
    ) -> str:
        return await gen_task_module(
            ctx,
            bound=bound,
            bound_ref=bound_ref,
            task=task,
            experiment_spec_id=experiment_spec_id,
            created_by=f"stage:{self.name}",
            catalog_id=catalog_id,
            feedback_ids=feedback_ids,
            context_id=context_id,
            is_repair=is_repair,
            slugs=slugs,
        )

    async def _gen_task_test(
        self,
        ctx: HarnessRunContext,
        *,
        bound_ref: PlanArtifactRef,
        task: object,
        slug: str,
        module_src: str,
        feedback_ids: list[str],
        context_id: str | None = None,
        wiring_id: str | None = None,  # noqa: ARG002
        is_repair: bool = False,
        bound: BoundWorkflow | None = None,
        slugs: dict[str, str] | None = None,
    ) -> str:
        return await gen_task_test(
            ctx,
            bound_ref=bound_ref,
            task=task,
            slug=slug,
            module_src=module_src,
            created_by=f"stage:{self.name}",
            feedback_ids=feedback_ids,
            context_id=context_id,
            is_repair=is_repair,
            bound=bound,
            slugs=slugs,
        )

    def _publish_cumulative(
        self,
        ctx: HarnessRunContext,
        *,
        bound: BoundWorkflow,
        bound_ref: PlanArtifactRef,
        slugs: dict[str, str],
        wf_files: dict[str, GeneratedFile],
        test_files: dict[str, GeneratedFile],
    ) -> None:
        # Only assemble tasks already written — mid-sequence imports of not-yet
        # generated modules would make pytest collection fail with ImportError.
        done_ids = {tid for tid, slug in slugs.items() if f"workflow/{slug}.py" in wf_files}
        partial_tasks = [t for t in bound.tasks if t.id in done_ids]
        partial_edges = [
            e for e in bound.edges if e.source_task_id in done_ids and e.target_task_id in done_ids
        ]
        partial = bound.model_copy(update={"tasks": partial_tasks, "edges": partial_edges})
        assembly = _render_assembly(partial, slugs)
        wf = WorkflowSource(
            source=assembly,
            module_name="workflow",
            bound_workflow_id=bound.id,
            symbols=("build_workflow",),
            files=list(wf_files.values()),
        )
        ctx.artifact_store.put_json(
            kind="workflow_source",
            obj=wf.model_dump(mode="json"),
            created_by=f"stage:{self.name}",
            parent_ids=[bound_ref.id],
        )
        # Minimal test_spec so validators downstream that require it stay happy.
        from molexp.harness.schemas import TestSpec, TestSpecBundle

        specs = []
        for task in bound.tasks:
            slug = slugs[task.id]
            if f"tests/test_{slug}.py" not in test_files:
                continue
            specs.append(
                TestSpec(
                    id=f"ts-{slug}",
                    name=f"test_{slug}",
                    kind="unit_test",
                    target_task_id=task.ir_task_id or task.id,
                    description=f"unit tests for {slug}",
                )
            )
        if not specs:
            t0 = bound.tasks[0]
            specs = [
                TestSpec(
                    id="ts-placeholder",
                    name="placeholder",
                    kind="unit_test",
                    target_task_id=t0.ir_task_id or t0.id,
                    description="placeholder",
                )
            ]
        bundle = TestSpecBundle(
            id=f"tsb-{bound.id}",
            bound_workflow_id=bound.id,
            specs=specs,
        )
        ts_ref = ctx.artifact_store.put_json(
            kind="test_spec",
            obj=bundle.model_dump(mode="json"),
            created_by=f"stage:{self.name}",
            parent_ids=[bound_ref.id],
        )
        test_src = TestSource(
            source="",
            module_name="test_suite",
            test_spec_id=bundle.id,
            bound_workflow_id=bound.id,
            files=list(test_files.values()),
        )
        ctx.artifact_store.put_json(
            kind="test_source",
            obj=test_src.model_dump(mode="json"),
            created_by=f"stage:{self.name}",
            parent_ids=[ts_ref.id, bound_ref.id],
        )

    async def _pytest_one(
        self, ctx: HarnessRunContext, *, test_path: str, task_id: str
    ) -> PlanArtifactRef:
        await self._materialize.run(ctx)
        return await self._run_pytest(ctx, targets=[test_path], label=task_id)

    async def _pytest_all(
        self, ctx: HarnessRunContext, *, test_files: dict[str, GeneratedFile]
    ) -> PlanArtifactRef:
        await self._materialize.run(ctx)
        return await self._run_pytest(ctx, targets=sorted(test_files.keys()), label="all-tasks")

    async def _run_pytest(
        self, ctx: HarnessRunContext, *, targets: list[str], label: str
    ) -> PlanArtifactRef:
        # DryRunExecutor plan tests skip real pytest; production LocalExecutor runs.
        return await run_pytest(
            ctx,
            executor=self._executor,
            targets=targets,
            cwd=str(ctx.workspace_root / "generated"),
            label=label,
            timeout_s=self._timeout_s,
            test_spec_id=f"sequential-{label}",
            created_by=f"stage:{self.name}",
            force_skip=self._force_skip_pytest,
            fail_label="sequential",
        )

    def _catalog_prompt(self, ctx: HarnessRunContext, bound_id: str) -> str | None:
        if ctx.capability_registry is None:
            return None
        catalog = render_capability_catalog(ctx.capability_registry.list_capabilities())
        return ctx.artifact_store.put_text(
            kind="capability_catalog",
            text=catalog,
            created_by=f"stage:{self.name}",
            parent_ids=[bound_id],
        ).id

    @staticmethod
    def _failure_text(ctx: HarnessRunContext, exc: StageExecutionError) -> str:
        """Prefer the full pytest traceback for repair / impl-vs-test classify."""
        return failure_text(ctx, exc)


def _topo_order(bound: BoundWorkflow) -> list:
    """Tasks in dependency order (edges source → target)."""
    ids = [t.id for t in bound.tasks]
    by_id = {t.id: t for t in bound.tasks}
    preds: dict[str, set[str]] = {i: set() for i in ids}
    for edge in bound.edges:
        if edge.target_task_id in preds and edge.source_task_id in by_id:
            preds[edge.target_task_id].add(edge.source_task_id)
    ordered: list = []
    remaining = set(ids)
    while remaining:
        ready = [i for i in remaining if not (preds[i] & remaining)]
        if not ready:
            # Cycle or missing — fall back to declaration order for the rest.
            ready = sorted(remaining)
        ready.sort()
        for i in ready:
            ordered.append(by_id[i])
            remaining.discard(i)
    return ordered


def _classify_failure(text: str) -> str:
    """Return ``impl`` if the stack points at workflow code, else ``test``."""
    # Real API mistakes in task modules (e.g. Frame.from_arrays) must rewrite
    # the module — not the pytest. Prefer the last traceback frame that is
    # under workflow/ vs tests/.
    wf_pos = max(text.rfind("workflow/"), text.rfind("workflow."))
    test_pos = max(text.rfind("tests/test_"), text.rfind("tests\\test_"))
    runtime_toks = (
        "BlockDtypeError",
        "unsupported dtype",
        "NameError",
        "AttributeError",
        "ImportError",
        "ModuleNotFoundError",
        "TypeError",
        "has no attribute",
    )
    if any(tok in text for tok in runtime_toks):
        if wf_pos >= 0 and wf_pos >= test_pos:
            return "impl"
        if "workflow/" in text or "workflow." in text or "in assemble_" in text:
            return "impl"
    if re.search(r"workflow/[\w/]+\.py", text) and any(
        t in text for t in ("NameError", "AttributeError", "TypeError", "has no attribute")
    ):
        if test_pos > wf_pos >= 0:
            return "test"
        return "impl"
    if "NameError" in text and "workflow" in text:
        return "impl"
    return "test"
