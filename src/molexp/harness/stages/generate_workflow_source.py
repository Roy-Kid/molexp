"""``GenerateWorkflowSource`` — emit molexp.workflow source, one call per task.

Fans workflow codegen out to **one ``workflow_source_file_writer`` call per
bound task**, run CONCURRENTLY, and synthesizes the ``workflow/__init__.py``
assembly deterministically from the bound workflow (imports + ``wf.task``
registration with its dependency edges) — no LLM guesses the wiring. One small
task per call keeps each pro codegen fast, lets it consult molmcp for just that
task, and sidesteps the whole-suite pro timeout; concurrency makes "many small
calls" cost the slowest single task, not their sum. A single-task workflow (or
an unparseable bound artifact) falls back to the original whole-suite
``workflow_source_writer`` call.

On a :class:`RepairLoop` retry the previous attempt's ``workflow_source_feedback``
(ReviewPlan / ValidateWorkflowSource findings) is threaded into every per-task
call so each task is corrected in place.
"""

from __future__ import annotations

import ast
import asyncio
import re
from typing import ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError
from molexp.harness.prompts.capability_catalog import render_capability_catalog
from molexp.harness.schemas import (
    AgentCallSpec,
    BoundWorkflow,
    PlanArtifactRef,
    WorkflowSource,
)
from molexp.harness.schemas.workflow_source import GeneratedFile
from molexp.harness.stages._resolve import feedback_inputs, require_latest

__all__ = ["GenerateWorkflowSource"]


class GenerateWorkflowSource(Stage):
    """Generate runnable molexp.workflow source, fanning one call per task."""

    name: ClassVar[str] = "generate_workflow_source"

    async def run(self, ctx: HarnessRunContext) -> PlanArtifactRef:
        if ctx.agent_gateway is None:
            raise StageExecutionError("GenerateWorkflowSource requires ctx.agent_gateway to be set")
        bound_ref = require_latest(ctx, "bound_workflow", stage=self.name)
        experiment_spec = require_latest(ctx, "experiment_spec", stage=self.name)
        feedback = feedback_inputs(ctx, "workflow_source_feedback")
        catalog_id = self._maybe_catalog(ctx, bound_ref.id)

        bound = self._parse_bound(ctx, bound_ref.id)
        if bound is None or len(bound.tasks) <= 1:
            # Single task (or unparseable) → the whole-suite call is small.
            spec = AgentCallSpec(
                agent_name="workflow_source_writer",
                input_artifact_ids=[bound_ref.id, experiment_spec.id, *feedback],
                prompt_artifact_id=catalog_id,
                output_schema=WorkflowSource.model_json_schema(),
            )
            return (await ctx.agent_gateway.call(spec)).output_artifact

        return await self._generate_per_task(
            ctx,
            bound_ref=bound_ref,
            bound=bound,
            experiment_spec_id=experiment_spec.id,
            catalog_id=catalog_id,
            feedback=feedback,
        )

    def _maybe_catalog(self, ctx: HarnessRunContext, bound_id: str) -> str | None:
        """Persist the grounded capability catalog as a prompt artifact (or None)."""
        if ctx.capability_registry is None:
            return None
        catalog = render_capability_catalog(ctx.capability_registry.list_capabilities())
        return ctx.artifact_store.put_text(
            kind="capability_catalog",
            text=catalog,
            created_by=f"stage:{self.name}",
            parent_ids=[bound_id],
        ).id

    def _parse_bound(self, ctx: HarnessRunContext, bound_id: str) -> BoundWorkflow | None:
        try:
            return BoundWorkflow.model_validate_json(ctx.artifact_store.get(bound_id))
        except Exception:
            return None

    async def _generate_per_task(
        self,
        ctx: HarnessRunContext,
        *,
        bound_ref: PlanArtifactRef,
        bound: BoundWorkflow,
        experiment_spec_id: str,
        catalog_id: str | None,
        feedback: list[str],
    ) -> PlanArtifactRef:
        """One concurrent ``workflow_source_file_writer`` call per bound task."""
        slugs = {task.id: _slug(task.ir_task_id or task.id) for task in bound.tasks}

        async def one(task) -> tuple[str, str]:
            """Return (bound_task_id, module_source) for one task."""
            slice_wf = bound.model_copy(
                update={"id": f"{bound.id}-{task.id}", "tasks": [task], "edges": []}
            )
            slice_ref = ctx.artifact_store.put_json(
                kind="bound_workflow_slice",
                obj=slice_wf.model_dump(mode="json"),
                created_by=f"stage:{self.name}",
                parent_ids=[bound_ref.id],
            )
            call = AgentCallSpec(
                agent_name="workflow_source_file_writer",
                input_artifact_ids=[slice_ref.id, experiment_spec_id, *feedback],
                prompt_artifact_id=catalog_id,
                output_schema=WorkflowSource.model_json_schema(),
            )
            result = await ctx.agent_gateway.call(call)
            partial = WorkflowSource.model_validate_json(
                ctx.artifact_store.get(result.output_artifact.id)
            )
            return task.id, _module_source(partial)

        pairs = await asyncio.gather(*(one(t) for t in bound.tasks))
        source_by_task = dict(pairs)

        # The function name is ENFORCED to the task's slug (not trusted from the
        # model), so the assembly's `from workflow.<slug> import <slug>` wiring
        # is deterministic and two tasks can never collide on one name.
        files: list[GeneratedFile] = []
        for task in bound.tasks:
            slug = slugs[task.id]
            src = _rename_top_function(source_by_task[task.id], slug)
            files.append(GeneratedFile(path=f"workflow/{slug}.py", source=src))

        assembly = _render_assembly(bound, slugs)
        assembled = WorkflowSource(
            source=assembly,
            module_name="workflow",
            bound_workflow_id=bound.id,
            symbols=("build_workflow",),
            files=files,
        )
        return ctx.artifact_store.put_json(
            kind="workflow_source",
            obj=assembled.model_dump(mode="json"),
            created_by=f"stage:{self.name}",
            parent_ids=[bound_ref.id, experiment_spec_id, *feedback],
        )


# ── pure helpers ────────────────────────────────────────────────────────────


def _slug(task_id: str) -> str:
    """A valid python module identifier derived from a task id."""
    token = re.sub(r"\W", "_", task_id).strip("_") or "task"
    if token[0].isdigit():
        token = f"t_{token}"
    return token


def _module_source(partial: WorkflowSource) -> str:
    """The one task module's text — its single file, else its ``source``."""
    if partial.files:
        return partial.files[0].source
    return partial.source


def _top_function_name(source: str) -> str | None:
    """Name of the module's first top-level (async) ``def``, or None."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    return None


def _rename_top_function(source: str, new_name: str) -> str:
    """Rename the module's single top-level (async) ``def`` to ``new_name``.

    The per-task module defines exactly one function and never references it by
    name (the engine calls it), so a precise rewrite of the ``def`` token is
    safe and preserves the body/formatting verbatim. Enforcing the name keeps
    the deterministic assembly wiring correct regardless of what the model
    chose. A no-op when the name already matches or the source won't parse.
    """
    original = _top_function_name(source)
    if original is None or original == new_name:
        return source
    return re.sub(
        rf"(\basync\s+def\s+|\bdef\s+){re.escape(original)}\s*\(",
        lambda m: f"{m.group(1)}{new_name}(",
        source,
        count=1,
    )


def _render_assembly(bound: BoundWorkflow, slugs: dict[str, str]) -> str:
    """Synthesize ``workflow/__init__.py`` from the bound tasks + edges.

    Each task's module + function are named by its slug; this imports them and
    registers them on a ``WorkflowCompiler`` with ``depends_on`` from the
    dependency edges, in topological order (the first registered task has no
    dependencies — the molexp contract).
    """
    deps: dict[str, list[str]] = {t.id: [] for t in bound.tasks}
    for edge in bound.edges:
        if edge.target_task_id in deps and edge.source_task_id in slugs:
            deps[edge.target_task_id].append(edge.source_task_id)
    order = _topo_order([t.id for t in bound.tasks], deps)

    lines = [
        "# workflow/__init__.py",
        "from molexp.workflow import WorkflowCompiler",
    ]
    for tid in order:
        lines.append(f"from workflow.{slugs[tid]} import {slugs[tid]}")
    lines += [
        "",
        "",
        "def build_workflow() -> WorkflowCompiler:",
        '    wf = WorkflowCompiler(name="plan_workflow")',
    ]
    for tid in order:
        dep_slugs = [slugs[d] for d in deps[tid]]
        if dep_slugs:
            dep_list = ", ".join(repr(d) for d in dep_slugs)
            lines.append(f"    wf.task(depends_on=[{dep_list}])({slugs[tid]})")
        else:
            lines.append(f"    wf.task({slugs[tid]})")
    lines.append("    return wf")
    return "\n".join(lines) + "\n"


def _topo_order(task_ids: list[str], deps: dict[str, list[str]]) -> list[str]:
    """Kahn topological sort; falls back to input order on a cycle."""
    remaining = {tid: set(deps.get(tid, [])) for tid in task_ids}
    order: list[str] = []
    while remaining:
        ready = [tid for tid in task_ids if tid in remaining and not remaining[tid]]
        if not ready:  # cycle — emit the rest in stable input order
            order.extend(tid for tid in task_ids if tid in remaining)
            break
        for tid in ready:
            order.append(tid)
            del remaining[tid]
            for d in remaining.values():
                d.discard(tid)
    return order
