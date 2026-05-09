"""Tests for control edges, route selection, Next/End sentinels, frontier scheduling.

Spec: .claude/specs/03-molexp-workflow-cycles.md
"""

from dataclasses import dataclass

import pydantic_graph
import pytest

from molexp.workflow import Actor, End, Task, WorkflowBuilder
from molexp.workflow.types import Next

# ── Sentinel imports ────────────────────────────────────────────────────────


def test_next_and_end_importable():
    """`Next` and `End` are public symbols on `molexp.workflow`."""
    assert Next is not None
    assert End is not None
    # Both are sentinel types (frozen dataclasses); instantiation works.
    assert Next("ok").label == "ok"
    assert isinstance(End(None), End)


def test_end_is_pydantic_graph_end():
    """`molexp.workflow.End` is a re-export of `pydantic_graph.End` — single
    source of truth, no duplicate sentinel (workflow-rectification §2)."""
    assert End is pydantic_graph.End


def test_task_does_not_inherit_basenode():
    """After the rectification, `Task` and `Actor` are plain abstract classes;
    only `WorkflowStep` inherits `pydantic_graph.BaseNode` (single-track)."""
    assert not issubclass(Task, pydantic_graph.BaseNode)
    assert not issubclass(Actor, pydantic_graph.BaseNode)


# ── Unconditional control edge ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unconditional_advances_frontier():
    """`wf.control(src, to)` alone advances the frontier from src to to.

    No `depends_on`; pure control-edge driven execution. Both tasks must run.
    """
    wf = WorkflowBuilder(name="unc-control")

    @wf.task
    async def alpha(ctx) -> str:
        return "alpha-out"

    @wf.task
    async def beta(ctx) -> str:
        return "beta-out"

    wf.entry("alpha")
    wf.control("alpha", "beta")

    result = await wf.build().execute()
    assert result.status == "completed"
    assert result.outputs == {"alpha": "alpha-out", "beta": "beta-out"}


# ── Branch + Next(label) ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_branch_routes_selected_by_next_label():
    """`Next("ok")` selects the route labelled `"ok"`, ignoring others."""
    wf = WorkflowBuilder(name="branch", entry="route")

    @wf.task(routes={"ok": "good", "fail": "bad"})
    async def route(ctx) -> Next:
        return Next("ok")

    @wf.task
    async def good(ctx) -> str:
        return "good-ran"

    @wf.task
    async def bad(ctx) -> str:
        return "bad-ran"

    result = await wf.build().execute()
    assert result.status == "completed"
    assert result.outputs.get("good") == "good-ran"
    assert "bad" not in result.outputs


# ── Loop via control edge ───────────────────────────────────────────────────


@dataclass(frozen=True)
class _Counter:
    n: int


@pytest.mark.asyncio
async def test_counter_loop_via_control_edge():
    """Counter that re-enters itself N times via control edge, then ends."""
    wf = WorkflowBuilder(name="counter", entry="tick")

    @wf.task(routes={"again": "tick", "done": "emit"})
    async def tick(ctx) -> tuple[_Counter, Next]:
        prev: _Counter | None = ctx.state.results.get("tick")
        n = (prev.n + 1) if prev else 1
        return _Counter(n=n), Next("again" if n < 3 else "done")

    @wf.task
    async def emit(ctx) -> int:
        return ctx.state.results["tick"].n

    result = await wf.build().execute()
    assert result.status == "completed"
    assert result.outputs["tick"] == _Counter(n=3)
    assert result.outputs["emit"] == 3


@pytest.mark.asyncio
async def test_self_loop_entry_accepted():
    """Entry task with self-loop control edge (counter --again--> counter, --done--> End) is legal."""
    wf = WorkflowBuilder(name="self-loop-entry", entry="counter")

    @wf.task(routes={"again": "counter", "done": "_end"})
    async def counter(ctx) -> tuple[_Counter, Next | End]:
        prev: _Counter | None = ctx.state.results.get("counter")
        n = (prev.n + 1) if prev else 1
        if n < 2:
            return _Counter(n=n), Next("again")
        return _Counter(n=n), End(None)

    # `routes` references "_end" sentinel target. We accept End(None) return short-circuiting it.
    # Compile must succeed even with self-loop entry.
    spec = wf.build()
    assert spec is not None  # compile didn't reject self-loop entry
    result = await spec.execute()
    assert result.status == "completed"
    assert result.outputs["counter"] == _Counter(n=2)


@pytest.mark.asyncio
async def test_loop_back_to_entry_accepted():
    """Entry node with control loop-back incoming edge (plan ↔ wait_approval) is legal."""
    wf = WorkflowBuilder(name="rework-loop", entry="plan")

    @wf.task
    async def plan(ctx) -> str:
        # Each iteration increments via state hack.
        prev = ctx.state.results.get("plan")
        return f"plan-v{(int(prev.split('v')[-1]) + 1) if prev else 1}"

    decisions = ["rework", "approve"]

    @wf.task(depends_on=["plan"], routes={"approve": "implement", "rework": "plan"})
    async def wait_approval(ctx) -> Next:
        d = decisions.pop(0)
        return Next(d)

    @wf.task(depends_on=["wait_approval"])
    async def implement(ctx) -> str:
        return "implemented"

    result = await wf.build().execute()
    assert result.status == "completed"
    assert result.outputs["plan"] == "plan-v2"  # ran twice
    assert result.outputs["implement"] == "implemented"


# ── End(None) semantics ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_end_is_frame_scoped():
    """`End(None)` is frame-scoped: same-frontier siblings still record their outputs."""
    wf = WorkflowBuilder(name="frame-end", entry="seed")

    @wf.task
    async def seed(ctx) -> int:
        return 0

    # Two siblings on the next frontier (parallel via control fan-out from seed).
    @wf.task(depends_on=["seed"])
    async def quitter(ctx) -> tuple[str, End]:
        return "quitter-out", End(None)

    @wf.task(depends_on=["seed"])
    async def survivor(ctx) -> str:
        return "survivor-out"

    result = await wf.build().execute()
    # Frontier-scoped End: both siblings ran in the same frontier and both got recorded.
    assert result.outputs["quitter"] == "quitter-out"
    assert result.outputs["survivor"] == "survivor-out"


@pytest.mark.asyncio
async def test_next_without_output_for_decision_node():
    """A decision-only node may return bare `Next(label)` — no output recorded."""
    wf = WorkflowBuilder(name="decision-only", entry="route")

    @wf.task(routes={"a": "leg_a", "b": "leg_b"})
    async def route(ctx) -> Next:
        return Next("a")

    @wf.task
    async def leg_a(ctx) -> str:
        return "took-a"

    @wf.task
    async def leg_b(ctx) -> str:
        return "took-b"

    result = await wf.build().execute()
    assert result.status == "completed"
    # Decision node didn't record an output.
    assert "route" not in result.outputs
    assert result.outputs.get("leg_a") == "took-a"


@pytest.mark.asyncio
async def test_value_then_next():
    """Returning `(value, Next(label))` records the value AND dispatches by label."""
    wf = WorkflowBuilder(name="value-and-next", entry="src")

    @wf.task(routes={"go": "dst"})
    async def src(ctx) -> tuple[int, Next]:
        return 42, Next("go")

    @wf.task
    async def dst(ctx) -> str:
        return "arrived"

    result = await wf.build().execute()
    assert result.outputs["src"] == 42
    assert result.outputs["dst"] == "arrived"


@pytest.mark.asyncio
async def test_value_then_end():
    """Returning `(value, End(None))` records the value AND terminates the workflow."""
    wf = WorkflowBuilder(name="value-and-end", entry="src")

    @wf.task
    async def src(ctx) -> tuple[int, End]:
        return 99, End(None)

    @wf.task
    async def never(ctx) -> str:  # dangling — should never execute
        return "should-not-run"

    wf.control("src", "never")

    result = await wf.build().execute()
    assert result.outputs["src"] == 99
    assert "never" not in result.outputs


# ── Actor with Next/End ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_actor_with_next():
    """An actor's async generator may `yield Next/End` as its terminating value."""
    wf = WorkflowBuilder(name="actor-next", entry="streamer")

    @wf.actor(routes={"emit": "sink"})
    async def streamer(ctx):
        for chunk in ["a", "b", "c"]:
            yield chunk
        yield Next("emit")  # terminal yield selects route

    @wf.task
    async def sink(ctx) -> str:
        return "sunk"

    result = await wf.build().execute()
    assert result.status == "completed"
    assert result.outputs["sink"] == "sunk"


# ── Route validation errors ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unknown_route_label_raises():
    """`Next("nonexistent")` raises `UnknownRouteError` listing declared labels."""
    from molexp.workflow import UnknownRouteError, WorkflowBuilder

    wf = WorkflowBuilder(name="bad-label", entry="route")

    @wf.task(routes={"a": "leg_a"})
    async def route(ctx) -> Next:
        return Next("nope")

    @wf.task
    async def leg_a(ctx) -> str:
        return "a"

    with pytest.raises(UnknownRouteError) as exc_info:
        await wf.build().execute()
    msg = str(exc_info.value)
    assert "nope" in msg
    assert "route" in msg  # task name
    assert "a" in msg  # declared labels listed


@pytest.mark.asyncio
async def test_branch_node_requires_next():
    """A branch-shaped node returning plain Output (no Next/End) raises `MissingRouteError`."""
    from molexp.workflow import MissingRouteError, WorkflowBuilder

    wf = WorkflowBuilder(name="missing-route", entry="route")

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
        await wf.build().execute()
    msg = str(exc_info.value)
    assert "route" in msg
    # declared labels listed
    assert "a" in msg and "b" in msg


# ── Join semantics ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_join_waits_for_all_data_deps():
    """A target with multi-`depends_on` waits in `pending_targets` until all data deps satisfy."""
    wf = WorkflowBuilder(name="join", entry="seed")

    @wf.task
    async def seed(ctx) -> int:
        return 1

    @wf.task(depends_on=["seed"])
    async def left(ctx) -> int:
        return ctx.inputs * 10

    @wf.task(depends_on=["seed"])
    async def right(ctx) -> int:
        return ctx.inputs * 100

    # collect joins: must wait for BOTH left and right.
    @wf.task(depends_on=["left", "right"])
    async def collect(ctx) -> int:
        # Multi-dep inputs come as dict[name → value].
        return ctx.inputs["left"] + ctx.inputs["right"]

    result = await wf.build().execute()
    assert result.outputs["collect"] == 110


# ── wf.loop primitive (spec 04 §4) ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_loop_overwrites_results():
    """Loop body re-runs on each iteration; ``results.<body>`` overwrites."""
    wf = WorkflowBuilder(name="loop-overwrite", entry="compute")

    counter = [0]

    @wf.task
    async def compute(ctx) -> int:
        counter[0] += 1
        return counter[0]

    @wf.task(depends_on=["compute"])
    async def check(ctx) -> Next:
        return Next("exit") if ctx.inputs >= 3 else Next("continue")

    wf.loop(body=["compute"], until="check", max_iters=10)

    result = await wf.build().execute()
    assert result.status == "completed"
    assert result.outputs["compute"] == 3
    assert counter[0] == 3


@pytest.mark.asyncio
async def test_loop_exit_advances_frontier():
    """``Next("exit")`` advances the frontier past the loop to the on_exit task."""
    wf = WorkflowBuilder(name="loop-exit", entry="seed")

    @wf.task
    async def seed(ctx) -> int:
        return 1

    iterations = [0]

    @wf.task(depends_on=["seed"])
    async def step(ctx) -> int:
        iterations[0] += 1
        return iterations[0]

    @wf.task(depends_on=["step"])
    async def gate(ctx) -> Next:
        return Next("exit") if iterations[0] >= 2 else Next("continue")

    @wf.task
    async def emit(ctx) -> str:
        return "emitted"

    wf.loop(body=["step"], until="gate", max_iters=10, on_exit="emit")

    result = await wf.build().execute()
    assert result.status == "completed"
    assert result.outputs["emit"] == "emitted"
    assert result.outputs["step"] == 2


@pytest.mark.asyncio
async def test_loop_max_iters_guard():
    """``max_iters`` forces ``Next("exit")`` and emits ``LoopMaxItersExceeded``.

    A body that always returns ``Next("continue")`` would otherwise loop
    forever; the guard caps iteration count at ``max_iters`` and emits a
    Python warning so callers can detect runaway loops without the
    workflow itself failing.
    """
    from molexp.workflow import LoopMaxItersExceeded, WorkflowBuilder

    wf = WorkflowBuilder(name="loop-runaway", entry="step")

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
        result = await wf.build().execute()

    assert result.status == "completed"
    assert runs[0] == 3


def test_loop_until_must_be_registered():
    """``wf.loop(until=...)`` referencing an unregistered task fails compile."""
    from molexp.workflow import UnknownTaskError, WorkflowBuilder

    wf = WorkflowBuilder(name="loop-bad-until", entry="step")

    @wf.task
    async def step(ctx) -> int:
        return 1

    wf.loop(body=["step"], until="nonexistent", max_iters=10)

    with pytest.raises(UnknownTaskError) as exc_info:
        wf.build()
    assert "nonexistent" in str(exc_info.value)


def test_loop_max_iters_must_be_positive():
    """``max_iters`` must be >= 1; 0 or negative is a programming error.

    Validated eagerly at ``wf.loop(...)`` call time, mirroring
    :meth:`Workflow.branch`'s shape-validation policy.
    """
    wf = WorkflowBuilder(name="loop-zero-iters", entry="step")

    @wf.task
    async def step(ctx) -> int:
        return 1

    @wf.task(depends_on=["step"])
    async def check(ctx) -> Next:
        return Next("exit")

    with pytest.raises(ValueError):
        wf.loop(body=["step"], until="check", max_iters=0)


# ── make_execution_id public API (spec 04 §6) ───────────────────────────────


def test_make_execution_id_public_export():
    """`make_execution_id` is exported from `molexp.workflow`."""
    from molexp.workflow import make_execution_id

    assert callable(make_execution_id)


def test_make_execution_id_no_run_id_returns_random_exec():
    """With run_id=None, returns `exec-{8 hex chars}`."""
    from molexp.workflow import make_execution_id

    eid = make_execution_id(run_id=None, run_dir=None)
    assert eid.startswith("exec-")
    suffix = eid.removeprefix("exec-")
    assert len(suffix) == 8
    int(suffix, 16)  # parses as hex


def test_make_execution_id_with_run_id_returns_base(tmp_path):
    """With a run_id but no prior execution directory, returns `exec-{run_id}`."""
    from molexp.workflow import make_execution_id

    eid = make_execution_id(run_id="abc123", run_dir=tmp_path)
    assert eid == "exec-abc123"


def test_make_execution_id_increments_on_existing_attempts(tmp_path):
    """Subsequent attempts add a `-N` suffix derived from existing dirs."""
    from molexp.workflow import make_execution_id

    exec_root = tmp_path / "executions"
    exec_root.mkdir()
    (exec_root / "exec-abc123").mkdir()

    eid = make_execution_id(run_id="abc123", run_dir=tmp_path)
    assert eid == "exec-abc123-2"


def test_submit_molq_plugins_do_not_reach_into_pydantic_graph():
    """ac-009 — `submit_molq` plugins must use the public `make_execution_id`."""
    import re
    from pathlib import Path

    plugin_dir = Path(__file__).resolve().parents[2] / "src" / "molexp" / "plugins"
    pattern = re.compile(r"_pydantic_graph")
    violations: list[str] = []
    for path in plugin_dir.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if pattern.search(line):
                violations.append(
                    f"{path.relative_to(plugin_dir.parent.parent.parent)}:{lineno}: {line.strip()}"
                )
    assert not violations, (
        "Plugins must not reach into molexp.workflow._pydantic_graph; "
        "use the public `from molexp.workflow import make_execution_id` instead.\n"
        "Violations:\n  " + "\n  ".join(violations)
    )
