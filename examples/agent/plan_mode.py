"""Interactive PlanMode demo wired to the molcrafts (``molmcp``) toolchain.

Two changes versus the default non-interactive run:

1. The user input embeds an explicit *Available molcrafts modules*
   preamble so the LLM prefers importing battle-tested ``molpy`` /
   ``molq`` / ``molexp`` APIs (CGSMILES parser, polymer builder,
   packmol packer, LAMMPS engine, scheduler submitor, …) over
   reimplementing simulation primitives from scratch in the generated
   task modules.
2. The default gate is replaced with the built-in
   :class:`~molexp.agent.modes.plan.PromptGatePolicy`: after the
   pipeline materializes every artifact, the user reviews each
   generated task (its test + implementation) one at a time and
   either approves it or marks it for replanning with a one-line
   feedback note. Rejected task ids flow into
   :class:`~molexp.agent.modes.plan.schemas.ApprovalDecision` as
   ``target_task_ids``; the repair loop then regenerates *only* those
   tasks on the next iteration (the rest are reused from disk),
   cascading downstream so the validation pass reruns.

To swap behavior, drop in
:class:`~molexp.agent.AutoApproveGatePolicy`
(``AutoApproveGatePolicy(ApprovalDecision(approved=True))``) for a
non-interactive run.

Run directly::

    python examples/agent/plan_mode.py

Set the provider's API key env var (e.g. ``DEEPSEEK_API_KEY`` for
DeepSeek, ``OPENAI_API_KEY`` for OpenAI) and adjust :data:`TIER_MODELS`
to point at real models. Each iteration does ~10-20 LLM calls; the
review prompts run between iterations.

The temporary workspace uses ``TemporaryDirectory(delete=False)`` so
the generated tree survives the script. The path is printed at the
end — ``cd`` there to inspect ``report/digest.md``,
``plan/implementation_plan.md``, ``ir/workflow.yaml``,
``ir/tasks/*.yaml``, ``src/experiment/...``, ``tests/*``,
``validation_report.md``, and ``manifest.yaml``.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import mollog

from molexp.agent import AgentRunner, AgentSession
from molexp.agent.modes import PlanMode
from molexp.agent.modes.plan import PlanWorkspaceHandle, PromptGatePolicy
from molexp.agent.router import ModelTier
from molexp.workspace import Workspace

mollog.configure(level="INFO")
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")
# Router already logs every LLM call with timing + token counts; the
# raw httpx "HTTP/1.1 200 OK" line is redundant noise. Silence it.
logging.getLogger("httpx").setLevel(logging.WARNING)

TIER_MODELS: dict[ModelTier, str] = {
    ModelTier.CHEAP: "deepseek:deepseek-v4-flash",
    ModelTier.DEFAULT: "deepseek:deepseek-v4-flash",
    ModelTier.HEAVY: "deepseek:deepseek-v4-pro",
}


MOLMCP_TOOLS_PREFACE = """\
Available molcrafts toolchain — prefer importing these over re-implementing
the same primitive yourself. Every module below ships on the system and is
also surfaced via the molmcp gateway; generated task modules should write
`from <path> import <name>` directly rather than re-deriving equations of
state, packing geometries, or LAMMPS input templates.

Molecular structure and topology
- molpy.parser.smiles.cgsmiles_parser   — parse CGSMILES strings into IR
- molpy.parser.smiles.converter_polymer — convert CGSMILES IR into a polymer
- molpy.builder.polymer.polymer_builder — assemble multi-residue chains
- molpy.builder.polymer.stochastic_generator — stochastic chain growth
- molpy.core.atomistic, molpy.core.cg   — Atomistic / CG entity containers
- molpy.core.box, molpy.core.region     — simulation box + region helpers

Packing
- molpy.pack.packer.packmol             — packmol-backed amorphous packing
- molpy.pack.constraint, molpy.pack.target — packing constraints / targets

LAMMPS I/O and execution
- molpy.io.emit.lammps                  — write LAMMPS data + input files
- molpy.io.data.lammps                  — read/write LAMMPS data files
- molpy.io.trajectory.lammps            — read LAMMPS trajectories
- molpy.io.log.lammps                   — parse LAMMPS log files
- molpy.engine.lammps                   — drive LAMMPS through Python

Force fields and typing
- molpy.io.forcefield.lammps            — LAMMPS forcefield I/O
- molpy.typifier.gaff, molpy.typifier.opls — atom-type assignment
- molpy.potential.{pair,bond,angle,dihedral,improper} — potential terms

Analysis
- molpy.compute.rdf                     — radial distribution function
- molpy.compute.pmsd                    — mean-squared displacement
- molpy.compute.time_series             — time-series helpers
- molpy.tool.polymer                    — polymer-specific analysis

Job submission and orchestration
- molq.submitor                         — submit + monitor scheduler jobs
- molq.scheduler                        — scheduler abstractions
- molexp.workspace.RunContext           — per-run scratch + assets
- molexp.workflow.Task / TaskContext    — task base classes

Set is_stub=true on a TaskIRBrief only when none of the modules above can
express the step. When in doubt, import a function from the list above and
let the runtime fail loudly at import time rather than re-implementing it.
"""


REPORT = """Molexp can automatically execute a reproducible simulation campaign for zwitterionic polymers with different charge topologies. Each polymer structure is defined using CGSMILES, where the relative placement of cationic and anionic groups is varied while keeping the chain length, number of chains, coarse-grained mapping, and simulation cell preparation protocol identical across all systems. This ensures that differences in the final properties mainly reflect structural topology rather than system-size effects.

For each CGSMILES-defined polymer, Molexp generates the molecular structure, builds chains of the same length, packs the same number of chains into an amorphous simulation box, and prepares LAMMPS input files. All systems first undergo the same equilibration workflow, including energy minimization, high-temperature relaxation, pressure equilibration, and production equilibration to obtain a well-relaxed reference configuration.

After equilibration, two independent LAMMPS workflows are launched. The first workflow estimates the glass transition temperature. The equilibrated system is cooled over a prescribed temperature range, and density or specific volume is recorded as a function of temperature. Tg is obtained by fitting the high-temperature and low-temperature regions separately and calculating the intersection of the two fitted lines. The second workflow evaluates mechanical properties by applying uniaxial deformation to the equilibrated configuration. The stress–strain curve is collected, and the Young's modulus is extracted from the initial linear elastic region.

Molexp then gathers all trajectories, logs, fitting parameters, and derived values into a unified result table. The final report compares Tg and modulus across zwitterionic topologies, includes uncertainty from repeated simulations where available, and stores all provenance information, including CGSMILES definitions, LAMMPS inputs, random seeds, raw outputs, fitting windows, and analysis scripts.
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


async def main() -> int:
    with TemporaryDirectory(delete=False) as tmp:
        workspace = Workspace(Path(tmp) / "ws")
        handle = PlanWorkspaceHandle.materialize(workspace, plan_id="demo")
        mode = PlanMode(
            workspace_handle=handle,
            gate_policy=PromptGatePolicy(),
            # 1 fresh pass + up to 3 replan rounds; the repair loop
            # raises RepairBudgetExceeded if the user never approves.
            max_iterations=4,
        )
        runner = AgentRunner(mode=mode, models=TIER_MODELS)
        session = AgentSession()
        # Embed the molcrafts toolchain preamble in the user input so it
        # rides through DraftReportDigest → DraftImplementationPlan →
        # CompileWorkflowIR and reaches the codegen system prompts.
        full_input = f"{MOLMCP_TOOLS_PREFACE}\n---\n\n{REPORT}"
        result = await runner.run(session, full_input)

        plan = (result.mode_state or {}).get("plan", {})

        print()
        print("=" * 72)
        print("Run finished")
        print("=" * 72)
        print(result.text)
        print()
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
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
