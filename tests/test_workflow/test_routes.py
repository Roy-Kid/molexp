"""Control-flow routing: control edges, ``Next``/``End`` sentinels, loops.

Graph-execution behaviors owned by ``molexp.workflow`` — the ``wf.control`` /
``routes=`` / ``wf.loop`` primitives lowered onto the engine and dispatched at
run time. Branch/loop happy-paths built from the *public* import surface live
in ``test_control_flow_public_api``; here we pin the engine-level routing
semantics (bare ``Next`` records no output, ``End`` termination/frame-scoping,
route validation errors), plus the ``make_execution_id`` id function and the
plugins→``_engine`` boundary lock.

Spec: .claude/specs/03-molexp-workflow-cycles.md
"""

from __future__ import annotations

import pytest

from molexp.workflow import (
    End,
    LoopMaxItersExceeded,
    MissingRouteError,
    Next,
    UnknownRouteError,
    UnknownTaskError,
    WorkflowCompiler,
    WorkflowRuntime,
    make_execution_id,
)


class TestControlFlowRouting:
    @pytest.mark.asyncio
    async def test_unconditional_control_edge_advances_frontier(self) -> None:
        """``wf.control(src, to)`` alone advances the frontier — no ``depends_on``."""
        wf = WorkflowCompiler(name="unc-control")

        @wf.task
        async def alpha(ctx) -> str:
            return "alpha-out"

        @wf.task
        async def beta(ctx) -> str:
            return "beta-out"

        wf.entry("alpha")
        wf.control("alpha", "beta")

        result = await WorkflowRuntime().execute(wf.compile())
        assert result.status == "succeeded"
        assert result.outputs == {"alpha": "alpha-out", "beta": "beta-out"}

    @pytest.mark.asyncio
    async def test_bare_next_routes_and_records_no_output(self) -> None:
        """A decision-only node returns bare ``Next(label)``: the labelled leg
        runs, the unrouted leg does not, and the node records no output."""
        wf = WorkflowCompiler(name="decision-only", entry="route")

        @wf.task(routes={"a": "leg_a", "b": "leg_b"})
        async def route(ctx) -> Next:
            return Next("a")

        @wf.task
        async def leg_a(ctx) -> str:
            return "took-a"

        @wf.task
        async def leg_b(ctx) -> str:
            return "took-b"

        result = await WorkflowRuntime().execute(wf.compile())
        assert result.status == "succeeded"
        assert "route" not in result.outputs  # decision node records no output
        assert result.outputs.get("leg_a") == "took-a"
        assert "leg_b" not in result.outputs  # unrouted leg never runs

    @pytest.mark.asyncio
    async def test_value_then_next_records_value_and_dispatches(self) -> None:
        """``(value, Next(label))`` records the value AND dispatches by label."""
        wf = WorkflowCompiler(name="value-and-next", entry="src")

        @wf.task(routes={"go": "dst"})
        async def src(ctx) -> tuple[int, Next]:
            return 42, Next("go")

        @wf.task
        async def dst(ctx) -> str:
            return "arrived"

        result = await WorkflowRuntime().execute(wf.compile())
        assert result.outputs["src"] == 42
        assert result.outputs["dst"] == "arrived"

    @pytest.mark.asyncio
    async def test_value_then_end_records_value_and_terminates(self) -> None:
        """``(value, End(None))`` records the value AND terminates downstream."""
        wf = WorkflowCompiler(name="value-and-end", entry="src")

        @wf.task
        async def src(ctx) -> tuple[int, End]:
            return 99, End(None)

        @wf.task
        async def never(ctx) -> str:  # dangling — should never execute
            return "should-not-run"

        wf.control("src", "never")

        result = await WorkflowRuntime().execute(wf.compile())
        assert result.outputs["src"] == 99
        assert "never" not in result.outputs

    @pytest.mark.asyncio
    async def test_end_is_frame_scoped(self) -> None:
        """``End(None)`` is frame-scoped: same-frontier siblings still record."""
        wf = WorkflowCompiler(name="frame-end", entry="seed")

        @wf.task
        async def seed(ctx) -> int:
            return 0

        @wf.task(depends_on=["seed"])
        async def quitter(ctx) -> tuple[str, End]:
            return "quitter-out", End(None)

        @wf.task(depends_on=["seed"])
        async def survivor(ctx) -> str:
            return "survivor-out"

        result = await WorkflowRuntime().execute(wf.compile())
        assert result.outputs["quitter"] == "quitter-out"
        assert result.outputs["survivor"] == "survivor-out"

    @pytest.mark.asyncio
    async def test_route_loops_back_to_entry_with_forwarded_value(self) -> None:
        """A routed edge back to the dep-less entry task is legal; the forwarded
        value re-delivers to the entry as ``prev`` (rework loop)."""
        wf = WorkflowCompiler(name="rework-loop", entry="plan")

        @wf.task
        async def plan(prev: str | None = None) -> str:
            return f"plan-v{(int(prev.split('v')[-1]) + 1) if prev else 1}"

        decisions = ["rework", "approve"]

        @wf.task(depends_on=["plan"], routes={"approve": "implement", "rework": "plan"})
        async def wait_approval(plan_value: str) -> tuple[str, Next]:
            d = decisions.pop(0)
            return plan_value, Next(d)

        @wf.task(depends_on=["wait_approval"])
        async def implement(approved_plan: str) -> str:
            return "implemented"

        result = await WorkflowRuntime().execute(wf.compile())
        assert result.status == "succeeded"
        assert result.outputs["plan"] == "plan-v2"  # ran twice
        assert result.outputs["implement"] == "implemented"

    @pytest.mark.asyncio
    async def test_actor_terminal_yield_selects_route(self) -> None:
        """An actor's async generator may ``yield Next(label)`` as its terminating value."""
        wf = WorkflowCompiler(name="actor-next", entry="streamer")

        @wf.actor(routes={"emit": "sink"})
        async def streamer(ctx):
            for chunk in ["a", "b", "c"]:
                yield chunk
            yield Next("emit")  # terminal yield selects route

        @wf.task
        async def sink(ctx) -> str:
            return "sunk"

        result = await WorkflowRuntime().execute(wf.compile())
        assert result.status == "succeeded"
        assert result.outputs["sink"] == "sunk"

    @pytest.mark.asyncio
    async def test_unknown_route_label_raises(self) -> None:
        """``Next("nope")`` raises ``UnknownRouteError`` listing declared labels."""
        wf = WorkflowCompiler(name="bad-label", entry="route")

        @wf.task(routes={"a": "leg_a"})
        async def route(ctx) -> Next:
            return Next("nope")

        @wf.task
        async def leg_a(ctx) -> str:
            return "a"

        with pytest.raises(UnknownRouteError) as exc_info:
            await WorkflowRuntime().execute(wf.compile())
        msg = str(exc_info.value)
        assert "nope" in msg
        assert "route" in msg  # task name
        assert "a" in msg  # declared labels listed

    @pytest.mark.asyncio
    async def test_branch_node_without_next_raises(self) -> None:
        """A branch-shaped node returning plain output raises ``MissingRouteError``."""
        wf = WorkflowCompiler(name="missing-route", entry="route")

        @wf.task(routes={"a": "leg_a", "b": "leg_b"})
        async def route(ctx) -> str:  # plain Output — illegal
            return "no-next-returned"

        @wf.task
        async def leg_a(ctx) -> str:
            return "a"

        @wf.task
        async def leg_b(ctx) -> str:
            return "b"

        with pytest.raises(MissingRouteError) as exc_info:
            await WorkflowRuntime().execute(wf.compile())
        msg = str(exc_info.value)
        assert "route" in msg
        assert "a" in msg and "b" in msg  # declared labels listed

    @pytest.mark.asyncio
    async def test_loop_max_iters_forces_exit_with_warning(self) -> None:
        """``max_iters`` forces ``Next("exit")`` and emits ``LoopMaxItersExceeded``
        rather than looping forever or failing the workflow."""
        wf = WorkflowCompiler(name="loop-runaway", entry="step")

        runs = [0]

        @wf.task
        async def step(ctx) -> int:
            runs[0] += 1
            return runs[0]

        @wf.task(depends_on=["step"])
        async def always_continue(ctx) -> Next:
            return Next("continue")

        wf.loop(body=["step"], until="always_continue", max_iters=3)

        with pytest.warns(LoopMaxItersExceeded):
            result = await WorkflowRuntime().execute(wf.compile())

        assert result.status == "succeeded"
        assert runs[0] == 3

    def test_loop_until_must_reference_registered_task(self) -> None:
        """``wf.loop(until=...)`` referencing an unregistered task fails compile."""
        wf = WorkflowCompiler(name="loop-bad-until", entry="step")

        @wf.task
        async def step(ctx) -> int:
            return 1

        wf.loop(body=["step"], until="nonexistent", max_iters=10)

        with pytest.raises(UnknownTaskError) as exc_info:
            wf.compile()
        assert "nonexistent" in str(exc_info.value)


class TestMakeExecutionId:
    def test_returns_base_id_without_prior_attempts(self, tmp_path) -> None:
        """With a run_id but no prior execution directory, returns ``exec-{run_id}``."""
        assert make_execution_id(run_id="abc123", run_dir=tmp_path) == "exec-abc123"

    def test_increments_suffix_over_existing_attempts(self, tmp_path) -> None:
        """A subsequent attempt adds a ``-N`` suffix derived from existing dirs."""
        exec_root = tmp_path / "executions"
        exec_root.mkdir()
        (exec_root / "exec-abc123").mkdir()
        assert make_execution_id(run_id="abc123", run_dir=tmp_path) == "exec-abc123-2"


def test_submit_molq_plugins_do_not_reach_into_engine() -> None:
    """ac-009 — plugins must use the public ``make_execution_id``, never
    reach into ``molexp.workflow._engine`` (architectural boundary lock)."""
    import re
    from pathlib import Path

    plugin_dir = Path(__file__).resolve().parents[2] / "src" / "molexp" / "plugins"
    pattern = re.compile(r"workflow[./]_engine")
    violations: list[str] = []
    for path in plugin_dir.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if pattern.search(line):
                violations.append(
                    f"{path.relative_to(plugin_dir.parent.parent.parent)}:{lineno}: {line.strip()}"
                )
    assert not violations, (
        "Plugins must not reach into molexp.workflow._engine; "
        "use the public `from molexp.workflow import make_execution_id` instead.\n"
        "Violations:\n  " + "\n  ".join(violations)
    )
