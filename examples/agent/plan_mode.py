"""``AgentRunner`` + ``PlanMode`` — real planning demo (molmcp + DeepSeek).

PlanMode turns a free-text research report into an approved, typed
:class:`~molexp.agent.modes._planning.PlanGraph`. You give it **one**
thing — a natural-language report — and it runs its internal seven-stage
pipeline (synthesize intent → clarify → explore capabilities → synthesize
candidates → select → preflight → approve), persisting every intermediate
artefact through a :class:`PlanFolder`. It writes no executable code.

Prerequisites:

* a provider API key — ``DEEPSEEK_API_KEY`` for the models below;
* the ``molmcp`` server on ``PATH`` — the ``ExploreCapabilities`` stage
  probes the molcrafts toolchain (molpy / molpack / lammps / …) through
  it. Without molmcp the probe falls back to a fail-closed null probe
  and the plan will not pass the ``capability_evidenced`` preflight.

Run directly::

    python examples/agent/plan_mode.py            # full campaign prompt
    python examples/agent/plan_mode.py --smoke    # short PEO-chain prompt
    python examples/agent/plan_mode.py --debug    # verbose router/probe logs
    python examples/agent/plan_mode.py --review   # prompt at the approval gate
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import mollog
from mollog import TextFormatter

from molexp.agent import AgentRunner, cli_ask
from molexp.agent.harness.events import ModeCompletedEvent
from molexp.agent.modes import PlanMode, PlanModeConfig
from molexp.agent.modes.plan import PlanFolder
from molexp.agent.router import ModelTier
from molexp.workspace import Workspace

_DEBUG = "--debug" in sys.argv[1:]
_SMOKE = "--smoke" in sys.argv[1:]
_REVIEW = "--review" in sys.argv[1:]

mollog.configure(
    level="DEBUG" if _DEBUG else "INFO",
    formatter=TextFormatter(template="{timestamp} - {message}", datefmt="%H:%M:%S"),
)

# Per-tier model map. DeepSeek exposes `deepseek-chat` (V3) and
# `deepseek-reasoner` (R1); adjust to whatever your key can reach.
TIER_MODELS: dict[ModelTier, str] = {
    ModelTier.CHEAP: "deepseek:deepseek-chat",
    ModelTier.DEFAULT: "deepseek:deepseek-chat",
    ModelTier.HEAVY: "deepseek:deepseek-reasoner",
}
PROBE_MODEL = "deepseek:deepseek-chat"


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
    """Return the prompt selected by the ``--smoke`` flag."""
    return SMOKE_TEST_REPORT if _SMOKE else REPORT


def _dump(title: str, path: Path) -> None:
    """Pretty-print one JSON artefact under a banner."""
    print(f"\n{'─' * 72}\n▶ {title}  ({path.name})\n{'─' * 72}")
    if not path.exists():
        print("  <not produced>")
        return
    print(json.dumps(json.loads(path.read_text()), indent=2, ensure_ascii=False))


async def main() -> int:
    # delete=False so the artefact tree survives for inspection.
    with TemporaryDirectory(delete=False) as tmp:
        workspace = Workspace(Path(tmp))
    plan_folder = cast(PlanFolder, workspace.add_folder(PlanFolder(name="demo")))

    mollog.info(f"plan run | plan_id={plan_folder.plan_id} workspace={workspace.root}")

    # PlanMode needs a probe_model + workspace so the molmcp-backed
    # capability probe is built for the ExploreCapabilities stage.
    mode = PlanMode(
        config=PlanModeConfig(max_repair_iterations=2),
        plan_folder=plan_folder,
        probe_model=PROBE_MODEL,
        workspace=workspace.root,
    )
    # `approval=` wires a ReviewPolicy into the harness `before_approval`
    # hook. `cli_ask` prompts the operator at the approve_direction gate;
    # `approval=None` (the default, no --review) auto-approves.
    runner = AgentRunner(
        mode=mode,
        models=TIER_MODELS,
        workspace=workspace.root,
        approval=cli_ask if _REVIEW else None,
    )
    session = runner.session("plan-demo")

    print("=" * 72)
    print(f"PlanMode — {'smoke (PEO chain)' if _SMOKE else 'full campaign'} prompt")
    print("=" * 72)
    completed: ModeCompletedEvent | None = None
    async for event in runner.run_events(session, _report()):
        kind = event.kind
        if kind == "stage_started":
            print(f"  ▶ {getattr(event, 'stage_name', '?')}")
        elif kind == "stage_completed":
            print(f"  ✔ {getattr(event, 'stage_name', '?')}")
        elif kind == "artifact_written":
            print(f"    · wrote {getattr(event, 'description', '')}")
        elif kind == "error":
            print(f"  ✗ ERROR: {getattr(event, 'message', '')}")
        elif isinstance(event, ModeCompletedEvent):
            completed = event
        else:
            print(f"  · {kind}")

    root = Path(str(plan_folder.path()))
    print("\n" + "=" * 72)
    print("STAGE ARTEFACTS")
    print("=" * 72)
    _dump("Stage 1  SynthesizeIntent     → IntentSpec", root / "intent.json")
    _dump("Stage 3  ExploreCapabilities  → CapabilityGraph", root / "capability_graph.json")
    for cand in sorted((root / "candidates").glob("*.json")):
        _dump(f"Stage 4  SynthesizeCandidates → candidate {cand.stem}", cand)
    _dump("Stage 5  SelectPlan           → selected PlanGraph", root / "selected_plan.json")
    _dump("Stage 6  PreflightPlanGraph   → PreflightReport", root / "preflight_report.json")

    assert completed is not None and completed.result is not None
    ms = completed.result.get("mode_state") or {}
    print("\n" + "=" * 72)
    print("RESULT")
    print("=" * 72)
    print(f"  text             : {completed.text}")
    print(f"  plan_state       : {ms.get('plan_state')}")
    print(f"  preflight_passed : {ms.get('preflight_passed')}")
    print(f"  has handoff      : {'handoff' in ms}")
    print(f"  artefacts at     : {root}")
    print(f"\n  cd {root}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
