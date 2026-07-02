"""Guarded workspace curation — relocate a run through the ChangeProposal gate.

molexp treats a destructive housekeeping action (moving a run, deleting a folder,
re-homing an asset) as a **high-risk mutation**: it never happens silently.
Instead it becomes a first-class, reviewable ``ChangeProposal`` that passes an
approval gate before anything on disk changes, and the proposal is preserved as
an audit record either way. This is the same backend the ``molexp curate`` CLI
and the ``POST /api/workspace/curate`` route use, so Python == UI.

This example, entirely offline (no LLM, no network):

1. builds a workspace with two experiments and one run,
2. proposes moving the run to the other experiment and drives it through the gate
   (auto-approved here; a TTY would prompt ``[y/N]``),
3. reads the §8 ``ChangeProposal`` record back from the audit trail so you can see
   the intent / reversibility / approval-level the gate decided on,
4. confirms the run actually moved on disk.

The gate stores its audit (the ``change_proposal`` artifact + ``harness.sqlite``)
under an auto-created ``curations`` project — a side effect worth knowing about.

Run directly::

    python examples/operations/curate.py
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import molexp as me
from molexp.harness.stages import auto_grant_approver
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.services.curate_runtime import build_curation_proposal, run_curation_proposal


async def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="molexp-curate-"))
    print(f"workspace root: {root}\n")

    # 1. A workspace with two experiments; the run 'expt-042' lives under 'run-a'.
    ws = me.Workspace(root, name="curate-demo")
    lab = ws.project("lab")
    lab.experiment("run-a").add_run(id="expt-042")
    lab.experiment("run-b")
    print("before:  run-a has", [r.id for r in lab.experiment("run-a").list_runs()])

    # 2. Propose moving the run to 'run-b' and drive it through the approval gate.
    #    build_curation_proposal turns the structured request into a §8 ChangeProposal;
    #    run_curation_proposal is the ONE backend the CLI + route also call.
    proposal = build_curation_proposal("move_run", run="expt-042", target_experiment="run-b")
    audit_run = ws.project("curations").experiment("curate").add_run(id="audit-1")
    result = await run_curation_proposal(
        proposal,
        workspace=ws,
        run=audit_run,
        approve=auto_grant_approver,  # a TTY would prompt [y/N]; here we auto-grant
    )
    outcome = result.execution_result
    print(f"\nproposal id: {result.id}")
    print(f"status:      {outcome.status if outcome else 'failed'}  (granted → executed)")

    # 3. The ChangeProposal is preserved as an audit artifact — read it back.
    store = FileArtifactStore(root=Path(audit_run.run_dir) / "artifacts")
    record = json.loads(store.get(store.list_by_kind("change_proposal")[0].id))
    print("\nthe §8 ChangeProposal the gate decided on:")
    print(f"  intent:         {record['intent']}")
    print(f"  op:             {record['proposed_change']['payload']['curation_op']}")
    print(f"  reversibility:  {record['reversibility']}")
    print(f"  approval_level: {record['approval_level']}")

    # 4. Confirm the mutation happened on disk (re-read the workspace fresh).
    fresh = me.Workspace.load(root).get_project("lab")
    print("\nafter:   run-a has", [r.id for r in fresh.get_experiment("run-a").list_runs()])
    print("         run-b has", [r.id for r in fresh.get_experiment("run-b").list_runs()])


if __name__ == "__main__":
    asyncio.run(main())
