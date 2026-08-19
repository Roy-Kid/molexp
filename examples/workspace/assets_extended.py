"""Error traces, checkpoint chaining, content-hash lookup, multi-scope asset queries.

Matches ``docs/guide/assets.md``.

Demonstrates:

1. ``ErrorTraceAsset`` — produced automatically on task failure.
2. Checkpoint chaining — ``parent_ckpt_id`` links sequential checkpoints.
3. ``scan.find_by_content_hash`` — resolve an asset from its hash.
4. ``DataAsset`` import with different actions — ``copy`` (default), ``symlink``.
5. ``scan.scan_assets`` with kind+scope filtering.
6. Multi-scope asset resolution — run → experiment → project → workspace.

Run directly::

    python examples/workspace/assets_extended.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import molexp as me
from molexp.workflow import WorkflowCompiler, WorkflowRuntime
from molexp.workspace.assets import scan

# ── Workflow with both success and failure paths ─────────────────────────────
wf = WorkflowCompiler(name="extended")

GOOD = True  # flip to False to see ErrorTraceAsset in action


@wf.task
def prepare(seed: int = 42) -> dict:
    return {"data": [1.0, 2.0, 3.0]}


@wf.task(depends_on=["prepare"])
def risky(data: list[float]) -> float:
    if not GOOD:
        raise ValueError("simulated failure — see ErrorTraceAsset")
    return sum(data)


compiled = wf.compile()


async def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="molexp-assets-ext-"))
    ws = me.Workspace(root, name="assets-ext-demo")

    # ── 1. Import data assets with different actions ─────────────────────
    external = root / "dataset.csv"
    external.write_text("x,y\n1,2\n3,4\n")
    ws.data_assets.import_asset("dataset-copy", external)  # default: copy
    ws.data_assets.import_asset("dataset-link", external, action="symlink")

    print(f"workspace root: {root}")

    # ── 2. Execute workflow — success path ───────────────────────────────
    exp = ws.project("demo").experiment("train").run(compiled, params={"seed": [42]})
    run = exp.list_runs()[0]
    with run.start() as ctx:
        result = await WorkflowRuntime().execute(compiled, run_context=ctx)

        # 3. Checkpoints with parent chaining
        ctx.checkpoint("epoch-1", data={"step": 1, "loss": 0.5})
        ctx.checkpoint("epoch-2", data={"step": 2, "loss": 0.2})
        ctx.checkpoint("epoch-3", data={"step": 3, "loss": 0.08})

        # 4. Artifacts and logs
        ctx.register_artifact(result.outputs, name="result.json")
        log = ctx.log("train")
        log.append("training complete")

    # ── 5. Content-hash lookup ───────────────────────────────────────────
    from molexp.ids import compute_content_hash

    test_file = root / "query-me.txt"
    test_file.write_text("hello reproducibility")
    file_hash = compute_content_hash(test_file)
    ws.data_assets.import_asset("query-target", test_file)
    found = scan.find_by_content_hash(ws.root, file_hash)
    print(f"\ncontent-hash lookup: {file_hash[:20]}...")
    if found:
        print(f"  found: {found.name} at {found.path}")

    # ── 6. Scan with kind filtering ──────────────────────────────────────
    all_assets = scan.scan_assets(ws.root)
    print(f"\ntotal assets: {len(all_assets)}")
    for asset in all_assets:
        kind = type(asset).__name__.removesuffix("Asset").lower()
        print(f"  [{kind:<15}] {asset.name:<25} scope={asset.scope.kind}")

    # ── 7. Multi-scope resolution (run → experiment → project → ws) ──────
    dataset = ctx.find_asset("dataset-copy")
    print(f"\nfind_asset 'dataset-copy': {dataset.path if dataset else 'not found'}")


if __name__ == "__main__":
    asyncio.run(main())
