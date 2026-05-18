"""Interactive PlanMode demo against the molcrafts (``molmcp``) toolchain.

Run directly::

    python examples/agent/plan_mode.py            # full interactive demo
    python examples/agent/plan_mode.py --smoke    # same flow, shorter prompt
    python examples/agent/plan_mode.py --debug    # include verbose router/node details

Set the provider's API key env var (e.g. ``DEEPSEEK_API_KEY`` for
DeepSeek, ``OPENAI_API_KEY`` for OpenAI) and adjust :data:`TIER_MODELS`
to point at real models. Each iteration does ~10-20 LLM calls; the
review prompts run between iterations.

The ``--smoke`` flag uses a short PEO-chain → LAMMPS-inputs prompt
intended to confirm the discovery node still reaches for the molcrafts
MCP rather than hand-rolling LAMMPS files.

The temporary workspace uses ``TemporaryDirectory(delete=False)`` so
the generated tree survives the script. The path is printed at the
end — ``cd`` there to inspect the artefacts.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import mollog
from mollog import TextFormatter

from molexp.agent import AgentRunner, AgentSession, HumanPolicy
from molexp.agent.modes import PlanMode
from molexp.agent.modes.plan import PlanFolder
from molexp.agent.router import ModelTier
from molexp.workspace import Workspace

_DEBUG = "--debug" in sys.argv[1:]
_SMOKE = "--smoke" in sys.argv[1:]

mollog.configure(
    level="DEBUG" if _DEBUG else "INFO",
    formatter=TextFormatter(template="{timestamp} - {message}", datefmt="%H:%M:%S"),
)

TIER_MODELS: dict[ModelTier, str] = {
    ModelTier.CHEAP: "deepseek:deepseek-v4-flash",
    ModelTier.DEFAULT: "deepseek:deepseek-v4-flash",
    ModelTier.HEAVY: "deepseek:deepseek-v4-pro",
}


REPORT = """Molexp can automatically execute a reproducible simulation campaign for zwitterionic polymers with different charge topologies. Each polymer structure is defined using CGSMILES, where the relative placement of cationic and anionic groups is varied while keeping the chain length, number of chains, coarse-grained mapping, and simulation cell preparation protocol identical across all systems. This ensures that differences in the final properties mainly reflect structural topology rather than system-size effects.

For each CGSMILES-defined polymer, Molexp generates the molecular structure, builds chains of the same length, packs the same number of chains into an amorphous simulation box, and prepares LAMMPS input files. All systems first undergo the same equilibration workflow, including energy minimization, high-temperature relaxation, pressure equilibration, and production equilibration to obtain a well-relaxed reference configuration.

After equilibration, two independent LAMMPS workflows are launched. The first workflow estimates the glass transition temperature. The equilibrated system is cooled over a prescribed temperature range, and density or specific volume is recorded as a function of temperature. Tg is obtained by fitting the high-temperature and low-temperature regions separately and calculating the intersection of the two fitted lines. The second workflow evaluates mechanical properties by applying uniaxial deformation to the equilibrated configuration. The stress-strain curve is collected, and the Young's modulus is extracted from the initial linear elastic region.

Molexp then gathers all trajectories, logs, fitting parameters, and derived values into a unified result table. The final report compares Tg and modulus across zwitterionic topologies, includes uncertainty from repeated simulations where available, and stores all provenance information, including CGSMILES definitions, LAMMPS inputs, random seeds, raw outputs, fitting windows, and analysis scripts.
"""


SMOKE_TEST_REPORT = """Build a single PEO (poly(ethylene oxide)) chain of 50 repeat units (-CH2-CH2-O-, hydroxyl-terminated) as an atomistic structure, assign OPLS-AA atom types, and emit LAMMPS inputs (a `data.peo` topology + coordinate file and a minimal `in.peo` script that performs energy minimization followed by a short NVT equilibration at 300 K).

Use the molcrafts toolchain wherever possible — the chain construction must produce a `molpy.Atomistic` object and the LAMMPS files must be written via `molpy.io.write_lammps_data` (or an equivalent `molpy.io.write_lammps_*` helper). Do not hand-roll the LAMMPS data file by formatting strings in Python; the point of the run is to confirm the planner reaches for `molpy` rather than reimplementing it.
"""


def _walk_artifacts(root: Path) -> list[tuple[Path, int]]:
    """Return every regular file under ``root`` as ``(path, size)``, sorted."""
    return sorted(
        ((p, p.stat().st_size) for p in root.rglob("*") if p.is_file()),
        key=lambda pair: pair[0].as_posix(),
    )


def _print_tree(root: Path) -> None:
    """Print every file under ``root`` with relative path + size."""
    artefacts = _walk_artifacts(root)
    if not artefacts:
        print("  (empty — pipeline did not materialize any files)")
        return
    width = max(len(p.relative_to(root).as_posix()) for p, _ in artefacts)
    for path, size in artefacts:
        rel = path.relative_to(root).as_posix()
        print(f"  {rel:<{width}}  {size:>8} B")


def _selected_report() -> str:
    """Return the prompt selected by CLI flags."""
    return SMOKE_TEST_REPORT if _SMOKE else REPORT


def _run_title() -> str:
    """Return the display title for the selected prompt."""
    return "Smoke prompt run finished" if _SMOKE else "Run finished"


async def main() -> int:
    with TemporaryDirectory(delete=False) as tmp:
        workspace = Workspace(Path(tmp))
        handle = cast(PlanFolder, workspace.add_folder(PlanFolder(name="demo")))

    mollog.info(f"starting plan run | plan_id={handle.plan_id} workspace={workspace.root}")

    mode = PlanMode(
        plan_folder=handle,
        step_policy=HumanPolicy(),
        final_policy=HumanPolicy(),
        max_iterations=4,
    )

    runner = AgentRunner(
        mode=mode,
        models=TIER_MODELS,
        workspace=workspace.root,
    )
    session = AgentSession()
    result = await runner.run(session, _selected_report())

    exec_id = handle.latest_execution_id()
    plan = (result.mode_state or {}).get("plan", {})

    print()
    print("=" * 72)
    print(_run_title())
    print("=" * 72)
    print(result.text)
    print()
    print(f"execution_id  : {exec_id}")
    print(f"approved      : {plan.get('approved')}")
    print(f"ready_for_run : {plan.get('ready_for_run')}")
    print(f"status        : {plan.get('status')}")
    print(f"workspace at  : {handle.root()}")
    print(f"manifest at   : {handle.manifest_path()}")

    print()
    print("=" * 72)
    print("Token usage:")
    print("=" * 72)
    print(result.usage_breakdown.render_table())

    print()
    print("=" * 72)
    print("Generated artefacts (open these in your editor to inspect):")
    print("=" * 72)
    _print_tree(handle.root())
    print()
    print(f"Tip: cd {handle.root()}")
    if exec_id:
        print(f"Resume: PlanMode.resume(execution_id={exec_id!r})")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
