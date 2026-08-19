"""``WorkflowRuntime.execute`` — the runtime API contract.

Post-rectification the runtime takes an opaque duck-typed ``run_context`` and a
``Mapping[str, Any]`` config — never a ``Workspace.Run`` or ``ProfileConfig``
(the legacy ``run=`` kwarg is gone; ``run_dir=`` accepts a path directly). This
file owns that boundary: failure→status, run_context handling (duck-typed, never
exposed on the public ``TaskContext``), executions materialization, the bare
``Runnable`` protocol body, and the ``scratch_root``/``ctx.workdir`` contract.

Graph topology (chains, diamonds, dict-merge binding, explicit parallelism) is
owned by ``test_parallel`` / ``test_by_name_binding`` / ``test_values_on_edges``
/ ``test_sync_tasks`` — not re-asserted here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.workflow import TaskContext, WorkflowCompiler, WorkflowRuntime


class _RunContextStub:
    """Minimal duck-typed ``run_context`` — what the runtime now requires.

    Exposes ``.run_dir`` / ``.config`` / ``.run`` so the runtime can extract a
    run dir and forward the value to its private channel. No ``Workspace`` import.
    """

    def __init__(
        self,
        *,
        run_dir: Path,
        config: dict | None = None,
        run_id: str | None = None,
        params: dict | None = None,
    ):
        self.run_dir = run_dir
        self.config = config or {}
        # Root-task params reach the body by name (engine reads run_context.params).
        self.params = params or {}
        self.run = type(
            "RunStub",
            (),
            {"id": run_id or "stub-run", "run_dir": run_dir},
        )()


@pytest.mark.asyncio
class TestWorkflowRuntimeExecute:
    async def test_task_failure_propagates_to_failed_status(self):
        wf = WorkflowCompiler(name="fail")

        @wf.task
        async def boom(ctx):
            raise RuntimeError("oops")

        result = await WorkflowRuntime().execute(wf.compile())
        assert result.status == "failed"

    async def test_external_runnable_protocol_body_executes(self):
        # A bare duck-typed body (no ``Task`` base — the ``Runnable`` protocol
        # surface) added via ``.add`` executes and its return becomes the output.
        class External:
            async def execute(self, ctx) -> int:
                return 99

        spec = WorkflowCompiler(name="ext").add(External(), name="ext").compile()
        result = await WorkflowRuntime().execute(spec)
        assert result.outputs["ext"] == 99

    async def test_run_context_is_not_exposed_on_task_ctx(self, tmp_path):
        # Pure-task-context contract: run_context is NOT forwarded to the public
        # TaskContext. A task accessing ctx.run_context raises AttributeError; the
        # engine still drives the run via its private channel.
        wf = WorkflowCompiler(name="no-run-context")

        run_ctx = _RunContextStub(
            run_dir=tmp_path / "run",
            config={"epochs": 1, "dataset": "md17"},
        )

        @wf.task
        async def inspect(ctx: TaskContext) -> bool:
            assert not hasattr(ctx, "run_context")
            return True

        result = await WorkflowRuntime().execute(wf.compile(), run_context=run_ctx)
        assert result.status == "succeeded"
        assert result.outputs["inspect"] is True

    async def test_duck_typed_run_context_needs_no_workspace_and_writes_executions(self, tmp_path):
        """The runtime drives a workflow with a stub run_context that has no
        Workspace ancestry whatsoever, and materializes workflow.json under
        run_dir/executions/<execution_id>/."""
        wf = WorkflowCompiler(name="duck")

        run_ctx = _RunContextStub(run_dir=tmp_path / "stub-run")

        @wf.task
        async def step(ctx: TaskContext) -> str:
            return "ok"

        result = await WorkflowRuntime().execute(wf.compile(), run_context=run_ctx)
        assert result.status == "succeeded"
        assert result.outputs["step"] == "ok"

        executions = run_ctx.run_dir / "executions"
        assert executions.exists(), "runtime must materialize executions/ under run_dir"
        wf_jsons = list(executions.rglob("workflow.json"))
        assert wf_jsons, "workflow.json must be written under run_dir/executions/<id>/"

    async def test_scratch_root_gives_every_task_a_workdir(self, tmp_path: Path) -> None:
        """``scratch_root=`` closes the ctx.workdir contract for bare executions.

        A plan-materialized driver runs ``WorkflowRuntime().execute(compiled,
        config=...)`` with NO tracked Run, but task bodies use ``ctx.workdir`` for
        scratch files. ``scratch_root`` mounts the content-addressed materialization
        store at an explicit location so ``ctx.workdir`` is never ``None``.
        """
        wf = WorkflowCompiler(name="scratch-demo")

        @wf.task
        async def write_report(ctx: TaskContext) -> dict:
            assert ctx.workdir is not None
            path = ctx.workdir / "report.json"
            path.write_text("{}")
            return {"report": str(path)}

        scratch = tmp_path / "scratch"
        result = await WorkflowRuntime().execute(wf.compile(), scratch_root=scratch)
        assert result.status == "succeeded"
        report = Path(result.outputs["write_report"]["report"])
        assert report.exists()
        assert scratch in report.parents

    async def test_without_scratch_root_keeps_workdir_none(self, tmp_path: Path) -> None:
        """No silent default: a bare execution that mounts no scratch_root keeps
        ``ctx.workdir`` as ``None`` (embedders must opt in explicitly — a cwd
        default would litter every caller's working directory)."""
        wf = WorkflowCompiler(name="no-scratch")

        @wf.task
        async def probe(ctx: TaskContext) -> dict:
            return {"workdir": ctx.workdir}

        result = await WorkflowRuntime().execute(wf.compile())
        assert result.status == "succeeded"
        assert result.outputs["probe"]["workdir"] is None
