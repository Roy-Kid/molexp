"""``AgentRunner`` + ``PlanMode`` + ``AuthorMode`` — Plan → Author chain demo.

This example chains PlanMode and AuthorMode end-to-end. PlanMode turns
a free-text research report into an approved typed
:class:`~molexp.agent.modes._planning.PlanGraph` (no executable code);
AuthorMode then lowers that plan into a real materialized workspace —
``ir/workflow.yaml`` + a Python package skeleton + per-task
implementations and tests — and runs each generated test in an
isolated subprocess via the source-grounded debug loop. The final
output is **concrete, runnable code** at
``<workspace>/plans/<plan_id>/experiment/...``.

The MCP-grounded paths (drafter, capability discovery, debug-loop
repair) all use molmcp through ``pydantic_ai.Agent(toolsets=[
MCPToolset(...)])`` so the LLM verifies project APIs against the real
source instead of guessing from training data.

Prerequisites:

* a provider API key — ``DEEPSEEK_API_KEY`` for the models below;
* the ``molmcp`` server on ``PATH`` (the seeded user-scope MCP server is
  what the drafter / grounding / repair agents attach to).

Run directly::

    python examples/agent/author_mode.py            # full campaign prompt
    python examples/agent/author_mode.py --smoke    # short PEO-chain prompt
    python examples/agent/author_mode.py --debug    # verbose logs
    python examples/agent/author_mode.py --review   # prompt at approval gates
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import cast

import mollog
from mollog import TextFormatter, Timer

from molexp.agent import AgentRunner, cli_ask
from molexp.agent.events import AgentEvent, ModeCompletedEvent
from molexp.agent.modes import (
    AuthorMode,
    AuthorModeConfig,
    PlanMode,
    PlanModeConfig,
)
from molexp.agent.modes.plan import ApprovedPlanHandoff, PlanFolder
from molexp.agent.router import ModelTier
from molexp.agent.types import utc_now
from molexp.workspace import Workspace

_DEBUG = "--debug" in sys.argv[1:]
_SMOKE = "--smoke" in sys.argv[1:]
_REVIEW = "--review" in sys.argv[1:]

mollog.configure(
    level="DEBUG" if _DEBUG else "INFO",
    formatter=TextFormatter(template="{timestamp} - {message}", datefmt="%H:%M:%S"),
)
# Ensure progress prints survive a buffered redirect (e.g. ``> log.txt``).
# Without this, a run that ends with an unhandled error inside an
# async generator can drop the last 4 KB of stdout (the timing summary).
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

# Per-tier model map. DeepSeek's current API exposes `deepseek-v4-flash`
# (general) and `deepseek-v4-pro` (premium). `deepseek-chat` /
# `deepseek-reasoner` are deprecated aliases (sunset 2026/07/24).
TIER_MODELS: dict[ModelTier, str] = {
    ModelTier.CHEAP: "deepseek:deepseek-v4-flash",
    ModelTier.DEFAULT: "deepseek:deepseek-v4-flash",
    ModelTier.HEAVY: "deepseek:deepseek-v4-pro",
}
# PlanMode's single research-and-plan agent (MCP-attached). Flash is fine
# — pydantic-ai drives the tool-call loop.
PLANNER_MODEL = "deepseek:deepseek-v4-flash"
# AuthorMode's source-grounded debug-loop repair model. Still benefits
# from the strong tier because it has to read project source.
REPAIR_MODEL = "deepseek:deepseek-v4-pro"


REPORT = """Molexp can automatically execute a reproducible simulation campaign for zwitterionic polymers with different charge topologies. Each polymer structure is defined using CGSMILES, where the relative placement of cationic and anionic groups is varied while keeping the chain length, number of chains, coarse-grained mapping, and simulation cell preparation protocol identical across all systems. This ensures that differences in the final properties mainly reflect structural topology rather than system-size effects.

For each CGSMILES-defined polymer, Molexp generates the molecular structure, builds chains of the same length, packs the same number of chains into an amorphous simulation box, and prepares LAMMPS input files. All systems first undergo the same equilibration workflow, including energy minimization, high-temperature relaxation, pressure equilibration, and production equilibration to obtain a well-relaxed reference configuration.

After equilibration, two independent LAMMPS workflows are launched. The first workflow estimates the glass transition temperature. The equilibrated system is cooled over a prescribed temperature range, and density or specific volume is recorded as a function of temperature. Tg is obtained by fitting the high-temperature and low-temperature regions separately and calculating the intersection of the two fitted lines. The second workflow evaluates mechanical properties by applying uniaxial deformation to the equilibrated configuration. The stress-strain curve is collected, and the Young's modulus is extracted from the initial linear elastic region.

Molexp then gathers all trajectories, logs, fitting parameters, and derived values into a unified result table. The final report compares Tg and modulus across zwitterionic topologies, includes uncertainty from repeated simulations where available, and stores all provenance information, including CGSMILES definitions, LAMMPS inputs, random seeds, raw outputs, fitting windows, and analysis scripts.
"""

SMOKE_TEST_REPORT = """Build a single PEO (poly(ethylene oxide)) chain of 50 repeat units (-CH2-CH2-O-, hydroxyl-terminated at both ends — HO-(CH2-CH2-O)50-H) as an atomistic structure, assign OPLS-AA atom types, and emit LAMMPS inputs: a `data.peo` topology + coordinate file and a minimal `in.peo` script that performs a brief energy minimization followed by a short NVT equilibration at 300 K (≈10 ps with a 1 fs timestep is fine; pick a reasonable thermostat).

Use the molcrafts toolchain (molpy and related packages) — discover the right APIs from the project source rather than inventing them, and do not hand-roll the LAMMPS data file by formatting strings in Python.

Assumptions you may rely on: the molcrafts toolchain (molpy and related packages) is already installed and importable in the execution environment, and LAMMPS is available; do not block planning on installation / environment-availability questions.
"""


def _report() -> str:
    return SMOKE_TEST_REPORT if _SMOKE else REPORT


def _handle_event(
    event: AgentEvent,
    starts: dict[str, datetime],
    elapsed: list[tuple[str, float]],
) -> None:
    """Render one event + record per-stage elapsed into ``elapsed``.

    Timings come from each event's own ``timestamp`` field, which the
    harness stamps at emission time. Reading wall-clock here would
    measure only the gap between drains in ``AgentRunner.run_events``
    (which buffers harness emissions until the mode next yields) — that
    produced bogus 0.01s readings for stages whose final yields landed
    after a sibling stage had already started.
    """
    kind = event.kind
    ts: datetime = getattr(event, "timestamp", utc_now())
    if kind == "stage_started":
        name = getattr(event, "stage_name", "?")
        starts[name] = ts
        print(f"  ▶ {name}", flush=True)
    elif kind == "stage_completed":
        name = getattr(event, "stage_name", "?")
        start_ts = starts.pop(name, None)
        seconds = (ts - start_ts).total_seconds() if start_ts is not None else 0.0
        elapsed.append((name, seconds))
        print(f"  ✔ {name}  [{seconds:.2f}s]", flush=True)
    elif kind == "artifact_written":
        print(f"    · wrote {getattr(event, 'description', '')}", flush=True)
    elif kind == "approval_requested":
        print(f"  ? approval: {getattr(event, 'gate', '?')}", flush=True)
    elif kind == "repair_proposed":
        print(f"  ↻ repair: {getattr(event, 'failed_invariant', '')}", flush=True)
    elif kind == "error":
        print(f"  ✗ ERROR: {getattr(event, 'message', '')}", flush=True)
    elif kind not in ("mode_started", "mode_completed", "approval_decided"):
        print(f"  · {kind}", flush=True)


def _print_timing_summary(label: str, total: float, elapsed: list[tuple[str, float]]) -> None:
    """Print a per-stage timing table + total."""
    print(f"\n  ── {label} timing summary ──")
    if not elapsed:
        print("    (no stages recorded)")
    else:
        width = max(len(name) for name, _ in elapsed)
        for name, secs in elapsed:
            pct = 100.0 * secs / total if total > 0 else 0.0
            print(f"    {name:<{width}}  {secs:7.2f}s  {pct:5.1f}%")
    print(f"    {'TOTAL':<{max(5, max((len(n) for n, _ in elapsed), default=5))}}  {total:7.2f}s")


async def _drive(
    runner: AgentRunner, session_id: str, label: str
) -> tuple[ModeCompletedEvent, float, list[tuple[str, float]]]:
    """Drive ``runner`` end-to-end; return terminal + total + per-stage elapsed."""
    session = runner.session(session_id)
    starts: dict[str, datetime] = {}
    elapsed: list[tuple[str, float]] = []
    terminal: ModeCompletedEvent | None = None
    with Timer(label, log=False) as total:
        async for event in runner.run_events(session, _report()):
            _handle_event(event, starts, elapsed)
            if isinstance(event, ModeCompletedEvent):
                terminal = event
    assert terminal is not None, "every AgentMode must yield a terminal ModeCompletedEvent"
    return terminal, total.elapsed, elapsed


def _dump_materialized_workspace(ws_path: Path) -> None:
    """Print the materialized layout + a peek at the generated workflow / code."""
    if not ws_path.is_dir():
        print(f"  <materialized workspace not found at {ws_path}>")
        return

    print(f"\n{'─' * 72}\n▶ Materialized workspace layout\n{'─' * 72}")
    for path in sorted(ws_path.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            rel = path.relative_to(ws_path)
            print(f"  {rel}  ({path.stat().st_size} bytes)")

    wf_yaml = ws_path / "ir" / "workflow.yaml"
    if wf_yaml.is_file():
        print(f"\n{'─' * 72}\n▶ ir/workflow.yaml\n{'─' * 72}")
        print(wf_yaml.read_text())

    src_dir = ws_path / "src" / "experiment" / "tasks"
    if src_dir.is_dir():
        impls = [p for p in sorted(src_dir.glob("*.py")) if p.name != "__init__.py"]
        if impls:
            first = impls[0]
            print(f"\n{'─' * 72}\n▶ first generated impl: {first.relative_to(ws_path)}\n{'─' * 72}")
            print(first.read_text())

    tests_dir = ws_path / "tests"
    if tests_dir.is_dir():
        tests = sorted(tests_dir.glob("test_*.py"))
        if tests:
            first = tests[0]
            print(f"\n{'─' * 72}\n▶ first generated test: {first.relative_to(ws_path)}\n{'─' * 72}")
            print(first.read_text())


async def main() -> int:
    # Write smoke artefacts under ``examples/agent/smoke_runs/<ts>/`` so
    # they can be reviewed in-tree instead of being scattered across
    # ``/var/folders/...``. Each run gets its own timestamped directory
    # — the parent ``smoke_runs/`` is gitignored so the artefacts don't
    # get committed by accident; delete the dir to clean up.
    runs_root = Path(__file__).resolve().parent / "smoke_runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    run_dir = runs_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir()
    workspace = Workspace(run_dir)
    # Point molmcp's MolexpProvider at this run's workspace so the
    # MCP server's ``molexp_list_projects`` tool resolves cleanly
    # instead of dumping a 30-line stack-trace banner every call.
    # Plumbed through the MCP-server env (per-subprocess) rather than
    # mutating ``os.environ`` of the parent — molmcp_env is forwarded
    # to ``StdioTransport(env=...)``.
    molmcp_env = {"MOLEXP_WORKSPACE": str(workspace.root)}
    plan_folder = cast(PlanFolder, workspace.add_folder(PlanFolder(name="demo")))

    mollog.info(f"author run | plan_id={plan_folder.plan_id} workspace={workspace.root}")

    # ── Stage A — PlanMode: free-text → approved typed PlanGraph ─────
    planner = PlanMode(
        config=PlanModeConfig(max_repair_iterations=2),
        plan_folder=plan_folder,
        planner_model=PLANNER_MODEL,
        molmcp_env=molmcp_env,
        workspace=workspace.root,
    )
    plan_runner = AgentRunner(
        mode=planner,
        models=TIER_MODELS,
        workspace=workspace.root,
        approval=cli_ask if _REVIEW else None,
    )
    print("=" * 72)
    print(f"PlanMode — {'smoke (PEO chain)' if _SMOKE else 'full campaign'}")
    print("=" * 72)
    plan_terminal, plan_total, plan_elapsed = await _drive(plan_runner, "plan-demo", "PlanMode")
    _print_timing_summary("PlanMode", plan_total, plan_elapsed)

    assert plan_terminal.result is not None
    plan_state = (plan_terminal.result.get("mode_state") or {}).get("plan_state")
    print(f"\nPlanMode final: plan_state={plan_state}")
    print(f"  artefacts at: {plan_folder.path()}")

    if plan_state != "approved":
        print("\n  Plan did not reach `approved` — cannot chain into AuthorMode.")
        print("  Inspect the plan artefacts above to see the failure mode.")
        print(f"\n  cd {plan_folder.path()}")
        return 0

    handoff_dump = (plan_terminal.result.get("mode_state") or {}).get("handoff")
    assert handoff_dump is not None, "approved plan must carry an ApprovedPlanHandoff"
    handoff = ApprovedPlanHandoff.model_validate(handoff_dump)

    # ── Stage B — AuthorMode: PlanGraph → materialized executable workspace ─
    author = AuthorMode(
        config=AuthorModeConfig(),
        plan_folder=plan_folder,
        handoff=handoff,
        repair_model=REPAIR_MODEL,  # source-grounded MCP-attached debug-loop repair
        workspace=workspace.root,
    )
    author_runner = AgentRunner(
        mode=author,
        models=TIER_MODELS,
        workspace=workspace.root,
        approval=cli_ask if _REVIEW else None,
    )
    print("\n" + "=" * 72)
    print("AuthorMode — materializing the approved plan")
    print("=" * 72)
    author_terminal, author_total, author_elapsed = await _drive(
        author_runner, "author-demo", "AuthorMode"
    )
    _print_timing_summary("AuthorMode", author_total, author_elapsed)
    print(f"\n  ▣ Plan + Author wall-clock: {plan_total + author_total:.2f}s")

    assert author_terminal.result is not None
    mode_state = author_terminal.result.get("mode_state") or {}
    author_state = mode_state.get("plan_state")
    mat_handoff = mode_state.get("handoff")
    print(f"\nAuthorMode final: plan_state={author_state}")
    print(f"  text: {author_terminal.text}")

    if mat_handoff:
        ws_path = Path(mat_handoff.get("experiment_workspace_path", ""))
        print(f"  materialized workspace: {ws_path}")
        _dump_materialized_workspace(ws_path)
        print("\n  To run the generated tests yourself:")
        print(f"    cd {ws_path} && PYTHONPATH=src pytest tests/ -q")
    else:
        print("  Materialization did not produce a handoff — see stage output above.")
        # Surface any captured codegen / validation failure so the operator
        # doesn't have to dig through the (silenced) stage internals.
        cg_err = getattr(author, "_codegen_error", None)
        if cg_err is not None:
            print(f"\n  CodegenError: {cg_err}")
            missing = getattr(cg_err, "missing", ())
            for m in missing:
                print(f"    · missing: {m.ref} — {m.detail}")
        vr = getattr(author, "_validation_report", None)
        if vr is not None and not getattr(vr, "ok", True):
            print("\n  WorkflowContract validation issues:")
            for issue in vr.issues:
                sev = getattr(issue, "severity", "?")
                msg = getattr(issue, "message", "")
                print(f"    [{sev}] {msg}")

    print(f"\n  cd {workspace.root}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
