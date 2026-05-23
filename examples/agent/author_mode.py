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
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import mollog
from mollog import TextFormatter

from molexp.agent import AgentRunner, cli_ask
from molexp.agent.harness.events import AgentEvent, ModeCompletedEvent
from molexp.agent.modes import (
    AuthorMode,
    AuthorModeConfig,
    PlanMode,
    PlanModeConfig,
)
from molexp.agent.modes.plan import ApprovedPlanHandoff, PlanFolder
from molexp.agent.router import ModelTier
from molexp.workspace import Workspace

_DEBUG = "--debug" in sys.argv[1:]
_SMOKE = "--smoke" in sys.argv[1:]
_REVIEW = "--review" in sys.argv[1:]

mollog.configure(
    level="DEBUG" if _DEBUG else "INFO",
    formatter=TextFormatter(template="{timestamp} - {message}", datefmt="%H:%M:%S"),
)

# Per-tier model map. Tweak to whatever your key reaches.
TIER_MODELS: dict[ModelTier, str] = {
    ModelTier.CHEAP: "deepseek:deepseek-chat",
    ModelTier.DEFAULT: "deepseek:deepseek-chat",
    ModelTier.HEAVY: "deepseek:deepseek-reasoner",
}
PROBE_MODEL = "deepseek:deepseek-chat"
REPAIR_MODEL = "deepseek:deepseek-chat"


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


def _print_event(event: AgentEvent) -> None:
    """Render one orchestration event as a single readable line."""
    kind = event.kind
    if kind == "stage_started":
        print(f"  ▶ {getattr(event, 'stage_name', '?')}")
    elif kind == "stage_completed":
        print(f"  ✔ {getattr(event, 'stage_name', '?')}")
    elif kind == "artifact_written":
        print(f"    · wrote {getattr(event, 'description', '')}")
    elif kind == "approval_requested":
        print(f"  ? approval: {getattr(event, 'gate', '?')}")
    elif kind == "repair_proposed":
        print(f"  ↻ repair: {getattr(event, 'failed_invariant', '')}")
    elif kind == "error":
        print(f"  ✗ ERROR: {getattr(event, 'message', '')}")
    elif kind not in ("mode_started", "mode_completed", "approval_decided"):
        print(f"  · {kind}")


async def _drive(runner: AgentRunner, session_id: str) -> ModeCompletedEvent:
    """Drive ``runner`` end-to-end and return the terminal ``ModeCompletedEvent``."""
    session = runner.session(session_id)
    terminal: ModeCompletedEvent | None = None
    async for event in runner.run_events(session, _report()):
        _print_event(event)
        if isinstance(event, ModeCompletedEvent):
            terminal = event
    assert terminal is not None, "every AgentMode must yield a terminal ModeCompletedEvent"
    return terminal


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
    # delete=False so the materialized workspace survives for inspection.
    with TemporaryDirectory(delete=False) as tmp:
        workspace = Workspace(Path(tmp))
    plan_folder = cast(PlanFolder, workspace.add_folder(PlanFolder(name="demo")))

    mollog.info(f"author run | plan_id={plan_folder.plan_id} workspace={workspace.root}")

    # ── Stage A — PlanMode: free-text → approved typed PlanGraph ─────
    planner = PlanMode(
        config=PlanModeConfig(max_repair_iterations=2),
        plan_folder=plan_folder,
        probe_model=PROBE_MODEL,
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
    plan_terminal = await _drive(plan_runner, "plan-demo")

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
    author_terminal = await _drive(author_runner, "author-demo")

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

    print(f"\n  cd {workspace.root}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
