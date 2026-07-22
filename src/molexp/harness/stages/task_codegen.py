"""Shared per-task codegen + pytest core for the task-realization stages.

Both :class:`~molexp.harness.stages.sequential_task_build.SequentialTaskBuild`
(the linear-pipeline build) and
:func:`~molexp.harness.stages.realize_task.realize_one_task` (the emergent
map-body) realize ONE :class:`BoundTask` the same way: generate its
``workflow/<slug>.py`` module and ``tests/test_<slug>.py`` unit test through the
injected ``AgentGateway`` (``use_mcp`` on repair turns), then run that one test
through the injected :class:`Executor`. This module owns those primitives so the
two callers share exactly one implementation.

The pytest runner is deliberately parameterized on ``cwd`` + ``targets``:
``SequentialTaskBuild`` runs the serial per-task pytest against the single
cumulative ``<run>/generated/`` tree (safe because it is serial), while the
concurrent realizer materializes each task into its OWN isolated
``<run>/realize/<slug>/`` tree so ``asyncio.gather``-launched workers never
clobber one another. Under a ``DryRunExecutor`` the real subprocess is skipped
entirely.

Import-guard note: like ``MaterializeExecution`` / ``CompileWorkflow`` this
module never imports ``molexp.workflow`` — the engine loads only inside the
executor subprocess, never in the harness process.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from mollog import get_logger

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.errors import StageExecutionError, StagePersistedFailureError
from molexp.harness.prompts.codegen_prompt import (
    assemble_task_codegen_prompt,
    assemble_task_test_prompt,
)
from molexp.harness.schemas import (
    AgentCallSpec,
    BoundWorkflow,
    CommandSpec,
    GeneratedFile,
    PlanArtifactRef,
    TestSource,
    WorkflowSource,
)
from molexp.harness.stages.generate_workflow_source import _module_source, _slug
from molexp.workspace.utils import generate_id

if TYPE_CHECKING:
    from molexp.harness.executors import Executor

__all__ = [
    "compose_task_markdown",
    "failure_text",
    "gen_task_module",
    "gen_task_test",
    "is_dry_run_executor",
    "materialize_task_sources",
    "run_isolated_task_pytest",
    "run_pytest",
]

_LOG = get_logger(__name__)


def is_dry_run_executor(executor: object) -> bool:
    """True when ``executor`` is a ``DryRunExecutor`` (per-task pytest is skipped)."""
    return type(executor).__name__ == "DryRunExecutor"


def _text_if(ctx: HarnessRunContext, art_id: str | None) -> str | None:
    if not art_id:
        return None
    try:
        return ctx.artifact_store.get(art_id).decode("utf-8")
    except Exception:
        return None


async def gen_task_module(
    ctx: HarnessRunContext,
    *,
    bound: BoundWorkflow,
    bound_ref: PlanArtifactRef,
    task: object,
    experiment_spec_id: str,
    created_by: str,
    catalog_id: str | None = None,
    feedback_ids: list[str] | None = None,
    context_id: str | None = None,
    is_repair: bool = False,
    slugs: dict[str, str] | None = None,
) -> str:
    """Generate one task's ``workflow`` module source (NOT yet slug-renamed).

    Fires a ``workflow_source_file_writer`` call whose YAML payload carries the
    frozen codegen contract, the bound task, the inter-task wiring, and any
    repair feedback. ``use_mcp`` follows ``is_repair`` — first pass runs
    catalog/compose-only, repair turns request molmcp grounding.
    """
    feedback_ids = list(feedback_ids or [])
    slug = _slug(getattr(task, "ir_task_id", None) or task.id)
    if slugs and task.id in slugs:
        slug = slugs[task.id]
    domain = _text_if(ctx, context_id)
    catalog = _text_if(ctx, catalog_id)
    feedback = "\n\n".join(_text_if(ctx, fid) or "" for fid in feedback_ids).strip() or None
    exp: dict | str | None
    try:
        raw = ctx.artifact_store.get(experiment_spec_id)
        exp = json.loads(raw.decode("utf-8"))
    except Exception:
        exp = _text_if(ctx, experiment_spec_id)
    yaml_prompt = assemble_task_codegen_prompt(
        task=task,
        slug=slug,
        bound=bound,
        slugs=slugs or {task.id: slug},
        experiment_spec=exp,
        domain_context=domain,
        feedback=feedback,
        catalog_excerpt=catalog,
    )
    prompt_ref = ctx.artifact_store.put_text(
        kind="codegen_prompt",
        text=yaml_prompt,
        created_by=created_by,
        parent_ids=[bound_ref.id],
    )
    call = AgentCallSpec(
        agent_name="workflow_source_file_writer",
        input_artifact_ids=[prompt_ref.id],
        prompt_artifact_id=None,
        output_schema=WorkflowSource.model_json_schema(),
        tier="heavy",
        use_mcp=is_repair,
    )
    result = await ctx.agent_gateway.call(call)  # type: ignore[union-attr]
    partial = WorkflowSource.model_validate_json(ctx.artifact_store.get(result.output_artifact.id))
    return _module_source(partial)


async def gen_task_test(
    ctx: HarnessRunContext,
    *,
    bound_ref: PlanArtifactRef,
    task: object,
    slug: str,
    module_src: str,
    created_by: str,
    feedback_ids: list[str] | None = None,
    context_id: str | None = None,
    is_repair: bool = False,
    bound: BoundWorkflow | None = None,
    slugs: dict[str, str] | None = None,
) -> str:
    """Generate one task's pytest module source given its (renamed) module.

    Fires a ``test_code_file_writer`` call. Also persists a
    ``workflow_source_file`` seed for lineage / downstream readers before
    asking for the test, matching the linear-build contract.
    """
    feedback_ids = list(feedback_ids or [])
    domain = _text_if(ctx, context_id)
    feedback = "\n\n".join(_text_if(ctx, fid) or "" for fid in feedback_ids).strip() or None
    yaml_prompt = assemble_task_test_prompt(
        task=task,
        slug=slug,
        module_source=module_src,
        bound=bound,
        slugs=slugs or {task.id: slug},
        domain_context=domain,
        feedback=feedback,
    )
    prompt_ref = ctx.artifact_store.put_text(
        kind="codegen_prompt",
        text=yaml_prompt,
        created_by=created_by,
        parent_ids=[bound_ref.id],
    )
    # Keep a workflow_source_file seed for lineage / downstream readers.
    ctx.artifact_store.put_json(
        kind="workflow_source_file",
        obj=WorkflowSource(
            source=module_src,
            module_name=slug,
            bound_workflow_id=bound_ref.id,
            symbols=(slug,),
            files=[GeneratedFile(path=f"workflow/{slug}.py", source=module_src)],
        ).model_dump(mode="json"),
        created_by=created_by,
        parent_ids=[bound_ref.id],
    )
    call = AgentCallSpec(
        agent_name="test_code_file_writer",
        input_artifact_ids=[prompt_ref.id],
        prompt_artifact_id=None,
        output_schema=TestSource.model_json_schema(),
        tier="heavy",
        use_mcp=is_repair,
    )
    result = await ctx.agent_gateway.call(call)  # type: ignore[union-attr]
    partial = TestSource.model_validate_json(ctx.artifact_store.get(result.output_artifact.id))
    if partial.files:
        return partial.files[0].source
    return partial.source


def failure_text(ctx: HarnessRunContext, exc: StageExecutionError) -> str:
    """Prefer the full pytest traceback for repair / impl-vs-test classify."""
    if isinstance(exc, StagePersistedFailureError):
        try:
            body = ctx.artifact_store.get(exc.persisted_ref.id).decode("utf-8")
        except Exception:
            return str(exc)
        # Legacy path: thin test_result JSON — attach latest feedback if any.
        if body.lstrip().startswith("{") and "pytest exit" in body:
            try:
                fb = ctx.artifact_store.latest_by_kind("test_code_feedback")
                if fb is not None:
                    extra = ctx.artifact_store.get(fb.id).decode("utf-8")
                    return f"{body}\n\n{extra}"
            except Exception:
                pass
        return body
    return str(exc)


async def run_pytest(
    ctx: HarnessRunContext,
    *,
    executor: Executor,
    targets: list[str],
    cwd: str,
    label: str,
    timeout_s: int,
    test_spec_id: str,
    created_by: str,
    force_skip: bool,
    fail_label: str = "task",
) -> PlanArtifactRef:
    """Run ``pytest`` over ``targets`` in ``cwd`` through ``executor``.

    Persists a ``test_result`` artifact, and on a nonzero exit persists the
    captured traceback as a ``test_code_feedback`` artifact before raising
    :class:`StagePersistedFailureError` (so repair sees the real stack, not the
    thin result JSON). Under a ``DryRunExecutor`` (``force_skip``) the real
    subprocess is skipped and a passing result is recorded.
    """
    if force_skip:
        return ctx.artifact_store.put_json(
            kind="test_result",
            obj={
                "id": f"test-result-{generate_id()}",
                "test_spec_id": test_spec_id,
                "status": "passed",
                "reason": "skipped_under_dry_run_executor",
                "targets": targets,
            },
            created_by=created_by,
            parent_ids=[],
        )
    cmd = [sys.executable, "-m", "pytest", *targets, "-q", "--import-mode=importlib"]
    if importlib.util.find_spec("pytest_asyncio") is not None:
        cmd += ["-o", "asyncio_mode=auto"]
    spec = CommandSpec(cmd=cmd, cwd=cwd, timeout_s=timeout_s)
    command = await executor.execute(spec, artifact_store=ctx.artifact_store)
    result_ref = ctx.artifact_store.put_json(
        kind="test_result",
        obj={
            "id": f"test-result-{generate_id()}",
            "test_spec_id": test_spec_id,
            "status": "passed" if command.exit_code == 0 else "failed",
            "reason": None if command.exit_code == 0 else f"pytest exit {command.exit_code}",
            "targets": targets,
        },
        created_by=created_by,
        parent_ids=[],
    )
    if command.exit_code != 0:
        out = _cmd_text(ctx, command)
        # Persist the *traceback* as the failure ref so repair classification
        # sees AttributeError/workflow paths — not the thin test_result JSON.
        fb_ref = ctx.artifact_store.put_text(
            kind="test_code_feedback",
            text=f"pytest failed for {label} ({targets}):\n\n{out}",
            created_by=created_by,
            parent_ids=[result_ref.id],
        )
        raise StagePersistedFailureError(
            fb_ref,
            f"{fail_label} pytest failed for {label!r} (exit {command.exit_code})",
        )
    return result_ref


async def run_isolated_task_pytest(
    ctx: HarnessRunContext,
    *,
    executor: Executor,
    slug: str,
    module_src: str,
    test_src: str,
    timeout_s: int,
    created_by: str,
) -> PlanArtifactRef:
    """Run one task's unit test against its OWN isolated source tree.

    Concurrency isolation: each task materializes into ``<run>/realize/<slug>/``
    (its own ``workflow/<slug>.py`` + ``tests/test_<slug>.py``), so
    ``asyncio.gather``-launched workers never race on the shared
    ``<run>/generated/`` tree that ``MaterializeExecution`` rewrites. Under a
    ``DryRunExecutor`` the pytest is skipped, so no tree is written.
    """
    force_skip = is_dry_run_executor(executor)
    task_dir = ctx.workspace_root / "realize" / slug
    if not force_skip:
        materialize_task_sources(task_dir, slug=slug, module_src=module_src, test_src=test_src)
    return await run_pytest(
        ctx,
        executor=executor,
        targets=[f"tests/test_{slug}.py"],
        cwd=str(task_dir),
        label=slug,
        timeout_s=timeout_s,
        test_spec_id=f"realize-{slug}",
        created_by=created_by,
        force_skip=force_skip,
    )


def materialize_task_sources(task_dir: Path, *, slug: str, module_src: str, test_src: str) -> None:
    """Write one task's isolated ``workflow/`` + ``tests/`` tree under ``task_dir``.

    The tree is wiped first (a repair attempt re-materializes) so a renamed or
    removed module from a prior attempt cannot linger and break collection.
    Sources get a lazy-annotations future import — the same guard
    ``MaterializeExecution`` applies — so bare molpy/molexp annotations imported
    inside the function body do not raise ``NameError`` at collection.
    """
    shutil.rmtree(task_dir, ignore_errors=True)
    wf_dir = task_dir / "workflow"
    tests_dir = task_dir / "tests"
    wf_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / "__init__.py").write_text("", encoding="utf-8")
    (wf_dir / f"{slug}.py").write_text(_with_future(module_src), encoding="utf-8")
    (tests_dir / f"test_{slug}.py").write_text(_with_future(test_src), encoding="utf-8")


def compose_task_markdown(task_text: str) -> str:
    """Best-effort in-process molmcp compose for one task (no stdio MCP)."""
    try:
        from molmcp.collection.browse import compose_context, open_ref, search_scoped
        from molmcp.config import load_config
        from molmcp.runtime import build_collection
    except Exception as exc:
        _LOG.debug(f"[task_codegen] molmcp compose unavailable: {exc!r}")
        return ""
    try:
        collection = build_collection(load_config())
    except Exception as exc:
        _LOG.debug(f"[task_codegen] build_collection failed: {exc!r}")
        return ""
    pages: list[str] = []
    try:
        pack = compose_context(collection, task=task_text, budget_chars=10_000)
        if pack.get("markdown"):
            pages.append(str(pack["markdown"]))
    except Exception as exc:
        _LOG.debug(f"[task_codegen] compose_context failed: {exc!r}")
    # Also open top search hits so writers see exact signatures (LAMMPS, pack, …).
    try:
        hits = search_scoped(collection, task_text, limit=6)
        for item in (hits.get("results") or [])[:4]:
            ref = item.get("ref") if isinstance(item, dict) else None
            if not ref:
                continue
            opened = open_ref(collection, str(ref))
            if opened.get("ok") and opened.get("markdown"):
                pages.append(str(opened["markdown"]))
    except Exception as exc:
        _LOG.debug(f"[task_codegen] open hits failed: {exc!r}")
    return "\n\n".join(pages)[:14_000]


def _with_future(source: str) -> str:
    if "from __future__ import annotations" in source:
        return source
    return "from __future__ import annotations\n" + source


def _cmd_text(ctx: HarnessRunContext, command: object) -> str:
    parts: list[str] = []
    for attr in ("stdout_artifact", "stderr_artifact"):
        ref = getattr(command, attr, None)
        if ref is None:
            continue
        rid = ref.id if hasattr(ref, "id") else ref
        try:
            parts.append(ctx.artifact_store.get(str(rid)).decode("utf-8", errors="replace"))
        except Exception:
            continue
    return "\n".join(parts) if parts else "(no captured output)"
