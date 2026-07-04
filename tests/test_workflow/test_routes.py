"""Tests for control edges, route selection, Next/End sentinels, frontier scheduling.

Spec: .claude/specs/03-molexp-workflow-cycles.md
"""

import pytest

from molexp.workflow import End, WorkflowCompiler, WorkflowRuntime
from molexp.workflow.types import Next

# ── Sentinel imports ────────────────────────────────────────────────────────


# ── Unconditional control edge ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unconditional_advances_frontier():
    """`wf.control(src, to)` alone advances the frontier from src to to.

    No `depends_on`; pure control-edge driven execution. Both tasks must run.
    """
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


# ── Branch + Next(label) ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_branch_routes_selected_by_next_label():
    """`Next("ok")` selects the route labelled `"ok"`, ignoring others."""
    wf = WorkflowCompiler(name="branch", entry="route")

    @wf.task(routes={"ok": "good", "fail": "bad"})
    async def route(ctx) -> Next:
        return Next("ok")

    @wf.task
    async def good(ctx) -> str:
        return "good-ran"

    @wf.task
    async def bad(ctx) -> str:
        return "bad-ran"

    result = await WorkflowRuntime().execute(wf.compile())
    assert result.status == "succeeded"
    assert result.outputs.get("good") == "good-ran"
    assert "bad" not in result.outputs


# ── Loop via control edge ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_loop_back_to_entry_accepted():
    """Entry node with control loop-back incoming edge (plan ↔ wait_approval) is legal."""
    wf = WorkflowCompiler(name="rework-loop", entry="plan")

    @wf.task
    async def plan(prev: str | None = None) -> str:
        # Values-on-edges: the rework loop-back delivers the previous plan
        # (forwarded by wait_approval) as the bound `prev`; None on first entry.
        return f"plan-v{(int(prev.split('v')[-1]) + 1) if prev else 1}"

    decisions = ["rework", "approve"]

    @wf.task(depends_on=["plan"], routes={"approve": "implement", "rework": "plan"})
    async def wait_approval(plan_value: str) -> tuple[str, Next]:
        d = decisions.pop(0)
        # Forward the plan value on the routed edge so a rework loop-back
        # re-delivers it to the dep-less entry task as its `prev` parameter.
        return plan_value, Next(d)

    @wf.task(depends_on=["wait_approval"])
    async def implement(approved_plan: str) -> str:
        return "implemented"

    result = await WorkflowRuntime().execute(wf.compile())
    assert result.status == "succeeded"
    assert result.outputs["plan"] == "plan-v2"  # ran twice
    assert result.outputs["implement"] == "implemented"


# ── End(None) semantics ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_end_is_frame_scoped():
    """`End(None)` is frame-scoped: same-frontier siblings still record their outputs."""
    wf = WorkflowCompiler(name="frame-end", entry="seed")

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

    result = await WorkflowRuntime().execute(wf.compile())
    # Frontier-scoped End: both siblings ran in the same frontier and both got recorded.
    assert result.outputs["quitter"] == "quitter-out"
    assert result.outputs["survivor"] == "survivor-out"


@pytest.mark.asyncio
async def test_next_without_output_for_decision_node():
    """A decision-only node may return bare `Next(label)` — no output recorded."""
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
    # Decision node didn't record an output.
    assert "route" not in result.outputs
    assert result.outputs.get("leg_a") == "took-a"


@pytest.mark.asyncio
async def test_value_then_next():
    """Returning `(value, Next(label))` records the value AND dispatches by label."""
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
async def test_value_then_end():
    """Returning `(value, End(None))` records the value AND terminates the workflow."""
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


# ── Actor with Next/End ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_actor_with_next():
    """An actor's async generator may `yield Next/End` as its terminating value."""
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


# ── Route validation errors ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unknown_route_label_raises():
    """`Next("nonexistent")` raises `UnknownRouteError` listing declared labels."""
    from molexp.workflow import UnknownRouteError, WorkflowCompiler

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
async def test_branch_node_requires_next():
    """A branch-shaped node returning plain Output (no Next/End) raises `MissingRouteError`."""
    from molexp.workflow import MissingRouteError, WorkflowCompiler

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
    # declared labels listed
    assert "a" in msg and "b" in msg


# ── Join semantics ──────────────────────────────────────────────────────────


# ── wf.loop primitive (spec 04 §4) ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_loop_overwrites_results():
    """Loop body re-runs on each iteration; ``results.<body>`` overwrites."""
    wf = WorkflowCompiler(name="loop-overwrite", entry="compute")

    counter = [0]

    @wf.task
    async def compute(ctx) -> int:
        counter[0] += 1
        return counter[0]

    @wf.task(depends_on=["compute"])
    async def check(count: int) -> Next:
        return Next("exit") if count >= 3 else Next("continue")

    wf.loop(body=["compute"], until="check", max_iters=10)

    result = await WorkflowRuntime().execute(wf.compile())
    assert result.status == "succeeded"
    assert result.outputs["compute"] == 3
    assert counter[0] == 3


@pytest.mark.asyncio
async def test_loop_max_iters_guard():
    """``max_iters`` forces ``Next("exit")`` and emits ``LoopMaxItersExceeded``.

    A body that always returns ``Next("continue")`` would otherwise loop
    forever; the guard caps iteration count at ``max_iters`` and emits a
    Python warning so callers can detect runaway loops without the
    workflow itself failing.
    """
    from molexp.workflow import LoopMaxItersExceeded, WorkflowCompiler

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


def test_loop_until_must_be_registered():
    """``wf.loop(until=...)`` referencing an unregistered task fails compile."""
    from molexp.workflow import UnknownTaskError, WorkflowCompiler

    wf = WorkflowCompiler(name="loop-bad-until", entry="step")

    @wf.task
    async def step(ctx) -> int:
        return 1

    wf.loop(body=["step"], until="nonexistent", max_iters=10)

    with pytest.raises(UnknownTaskError) as exc_info:
        wf.compile()
    assert "nonexistent" in str(exc_info.value)


# ── make_execution_id public API (spec 04 §6) ───────────────────────────────


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


def test_submit_molq_plugins_do_not_reach_into_engine():
    """ac-009 — `submit_molq` plugins must use the public `make_execution_id`."""
    import re
    from pathlib import Path

    plugin_dir = Path(__file__).resolve().parents[2] / "src" / "molexp" / "plugins"
    pattern = re.compile(r"workflow[./]_engine")
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
        "Plugins must not reach into molexp.workflow._engine; "
        "use the public `from molexp.workflow import make_execution_id` instead.\n"
        "Violations:\n  " + "\n  ".join(violations)
    )
