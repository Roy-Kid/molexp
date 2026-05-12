"""Interactive PlanMode demo against the molcrafts (``molmcp``) toolchain.

Out of the box this script demonstrates two non-default behaviours:

1. **Auto-discovered molcrafts toolchain.** No tool list is hard-coded in
   the user input. Instead:

   * On first :class:`~molexp.agent.mcp.store.McpStore` construction the
     default ``molmcp`` stdio entry is seeded into ``~/.molexp/mcp.json``
     (see :mod:`molexp.agent.mcp.defaults`).
   * :class:`~molexp.agent.AgentRunner` auto-prepends the seeded
     ``usage_instructions`` (a short pointer to the ``molmcp__*`` tools)
     to the mode's system prompt.
   * PlanMode's discovery node drives a pydantic-ai agent attached to
     ``molmcp`` via ``Agent(toolsets=[MCPServerStdio(...)])`` and uses
     ``list_modules`` / ``list_symbols`` / ``search_source`` /
     ``get_signature`` / ``get_docstring`` to pull concrete symbol
     references into the codegen context.

   Passing ``workspace=workspace.root`` to the runner also lets a
   per-workspace ``.mcp.json`` override the user-scope entry. Make sure
   ``molmcp`` is installed and resolvable on ``$PATH`` (or set
   ``MOLEXP_MOLMCP_COMMAND`` before the first run to point at a custom
   binary path).

2. **Two review hooks, both interactive.** PlanMode now exposes two
   independently configurable :class:`~molexp.agent.ReviewPolicy`
   slots:

   * ``step_policy`` — fires after every non-terminal node's
     ``_execute`` (digest, plan brief, IR compile, codegen, validation).
     This demo plugs in :class:`~molexp.agent.HumanPolicy` so the CLI
     prompts you after each step ("approve DraftReportDigest? [y/N/?]");
     ``n`` triggers a repair iteration of that step plus its downstream
     cascade.
   * ``final_policy`` — fires once at the ``HumanReview`` node after the
     whole pipeline materializes.  Same :class:`HumanPolicy`, but the
     bundled :func:`~molexp.agent.cli_ask` callback detects the
     plan-final view and walks task-by-task through every generated
     task (test + implementation) so you can approve / reject each one
     independently.  Rejected task ids cascade into the next iteration's
     codegen via :attr:`ReviewDecision.target_task_ids`.

To skip a hook entirely, pass :class:`~molexp.agent.BypassPolicy`.  For
example, ``step_policy=BypassPolicy()`` would silence the per-step
prompts and only ask once at the end.

Run directly::

    python examples/agent/plan_mode.py            # full interactive demo
    python examples/agent/plan_mode.py --smoke    # non-interactive smoke test
    python examples/agent/plan_mode.py --debug    # include verbose router/node details

The script runs a preflight check before any LLM calls: provider key,
``pydantic-ai`` availability, ``molmcp`` MCP config, executable lookup,
and a stdio MCP handshake.  Pass ``--skip-preflight`` only when you are
intentionally debugging a failing environment.

Set the provider's API key env var (e.g. ``DEEPSEEK_API_KEY`` for
DeepSeek, ``OPENAI_API_KEY`` for OpenAI) and adjust :data:`TIER_MODELS`
to point at real models. Each iteration does ~10-20 LLM calls; the
review prompts run between iterations.

The ``--smoke`` mode runs a short PEO-chain → LAMMPS-inputs prompt
with :class:`~molexp.agent.BypassPolicy` on both review hooks and a
shrunk repair budget; it is the fast, hands-off variant of the demo
intended to confirm the discovery node still reaches for the
molcrafts MCP rather than hand-rolling LAMMPS files. Inspect the
generated tree afterwards to see what the planner produced.

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

from molexp.agent import AgentRunner, AgentSession, BypassPolicy, HumanPolicy
from molexp.agent.modes import PlanMode
from molexp.agent.modes.plan import PlanWorkspaceHandle
from molexp.agent.modes.plan.preflight import check_plan_runtime
from molexp.agent.router import ModelTier
from molexp.workspace import Workspace

_DEBUG = "--debug" in sys.argv[1:]
_SKIP_PREFLIGHT = "--skip-preflight" in sys.argv[1:]

mollog.configure(level="DEBUG" if _DEBUG else "INFO")
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")
# Router already logs every LLM call with timing + token counts; the
# raw httpx "HTTP/1.1 200 OK" line is redundant noise. Silence it.
logging.getLogger("httpx").setLevel(logging.WARNING)

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


async def _preflight_or_exit(workspace: Workspace) -> int | None:
    """Run environment checks before spending LLM calls."""
    if _SKIP_PREFLIGHT:
        print("PlanMode preflight skipped (--skip-preflight).")
        return None

    report = await check_plan_runtime(
        workspace=workspace.root,
        models=TIER_MODELS,
        require_molmcp=True,
        verify_molmcp_stdio=True,
    )
    print(report.render())
    if report.passed:
        print()
        return None
    print()
    print("Fix the failed preflight check(s), or pass --skip-preflight to run anyway.")
    return 2


async def main() -> int:
    with TemporaryDirectory(delete=False) as tmp:
        workspace = Workspace(Path(tmp) / "ws")
        preflight_status = await _preflight_or_exit(workspace)
        if preflight_status is not None:
            return preflight_status
        handle = PlanWorkspaceHandle.materialize(workspace, plan_id="demo")
        mode = PlanMode(
            workspace_handle=handle,
            step_policy=HumanPolicy(),  # CLI prompt after every non-terminal node
            final_policy=HumanPolicy(),  # CLI walk task-by-task at the plan-final review
            # 1 fresh pass + up to 3 replan rounds; the repair loop
            # raises RepairBudgetExceeded if the user never approves.
            max_iterations=4,
        )
        # Pass workspace=workspace.root so the runner's MCP lookup also
        # sees a per-workspace .mcp.json (when present); the user-scope
        # ~/.molexp/mcp.json — including the auto-seeded ``molmcp`` —
        # is read regardless.
        runner = AgentRunner(
            mode=mode,
            models=TIER_MODELS,
            workspace=workspace.root,
        )
        session = AgentSession()
        result = await runner.run(session, REPORT)

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


async def smoke_test() -> int:
    """Non-interactive PEO-chain → LAMMPS-inputs smoke run.

    Mirrors :func:`main` but swaps both review hooks for
    :class:`~molexp.agent.BypassPolicy` and shrinks the repair budget
    to two iterations so the script runs end-to-end without a human
    at the keyboard. The generated tree is left on disk for manual
    inspection.
    """
    with TemporaryDirectory(delete=False) as tmp:
        workspace = Workspace(Path(tmp) / "ws")
        preflight_status = await _preflight_or_exit(workspace)
        if preflight_status is not None:
            return preflight_status
        handle = PlanWorkspaceHandle.materialize(workspace, plan_id="smoke")
        mode = PlanMode(
            workspace_handle=handle,
            step_policy=BypassPolicy(),
            final_policy=BypassPolicy(),
            # 1 fresh pass + 1 replan; smoke test is meant to be fast.
            max_iterations=2,
        )
        runner = AgentRunner(
            mode=mode,
            models=TIER_MODELS,
            workspace=workspace.root,
        )
        session = AgentSession()
        result = await runner.run(session, SMOKE_TEST_REPORT)

        plan = (result.mode_state or {}).get("plan", {})

        print()
        print("=" * 72)
        print("Smoke test finished")
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
    entry = smoke_test if "--smoke" in sys.argv[1:] else main
    sys.exit(asyncio.run(entry()))
