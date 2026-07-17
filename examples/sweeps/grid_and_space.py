"""Grid and random parameter sweeps — RunSet, RunSetResult, and idempotent re-declaration.

Matches ``docs/guide/sweeps.md``.

Demonstrates:

1. ``GridSpace`` — exhaustive Cartesian product over parameter values.
2. Dict shorthand — ``sweep(wf, {"lr": [1e-3, 1e-4]})`` is a GridSpace.
3. ``RunSet.execute(parallel=N)`` — concurrent execution.
4. ``RunSetResult.to_records()`` — one flat dict per run.
5. ``RunSetResult.min_by(key)`` / ``max_by(key)`` — extreme record lookup.
6. ``UniformSpace`` — random sampling from discrete value lists.
7. Reading back a finished sweep with ``exp.runs().collect()``.
8. Idempotent re-declaration — same params → same run ids.

Run directly::

    python examples/sweeps/grid_and_space.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import molexp as me
from molexp.workflow import WorkflowCompiler
from molexp.workspace.param import GridSpace, UniformSpace

wf = WorkflowCompiler(name="train")


@wf.task
def model(lr: float = 1e-3) -> dict:
    """Root task — receives sweep params by name."""
    return {"loss": lr * 100, "accuracy": 1.0 - lr}


@wf.task(depends_on=["model"])
def report(loss: float, accuracy: float) -> float:
    return loss


compiled = wf.compile()


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="molexp-sweeps-"))
    ws = me.Workspace(root, name="sweeps-demo")

    # ── 1. GridSpace — exhaustive Cartesian product ─────────────────────
    print("── GridSpace sweep ──────────────────────────────────────")
    exp = ws.project("demo").experiment("lr-scan")
    space = GridSpace({"lr": [1e-3, 5e-4, 1e-4], "seed": [42]})
    scan = exp.sweep(wf, params=space)
    summary = scan.execute(parallel=2)

    for row in summary.to_records():
        print(f"  lr={row['lr']:.0e}  status={row['status']}  loss={row['report']:.4f}")

    best = summary.min_by("report")
    print(f"  best lr={best['lr']:.0e}  run_id={best['run_id']}")

    # ── 2. Dict shorthand — same as GridSpace ──────────────────────────
    print()
    print("── Dict shorthand ───────────────────────────────────────")
    quick = ws.project("demo").experiment("quick-sweep").sweep(wf, {"lr": [1e-2]})
    quick_result = quick.execute()
    print(f"  runs: {len(quick_result)}")

    # ── 3. Idempotent re-declaration — same params → same runs ─────────
    print()
    print("── Idempotent re-declaration ────────────────────────────")
    again = exp.sweep(wf, params=space)
    assert len(again) == len(scan), "re-declaring the same sweep must produce the same run count"
    print(f"  first declaration:  {len(scan)} runs")
    print(f"  second declaration: {len(again)} runs (same ids)")

    # ── 4. Read back finished sweep ────────────────────────────────────
    print()
    print("── Read back from disk ──────────────────────────────────")
    collected = exp.runs().collect()
    print(f"  collected {len(collected)} runs")
    for row in collected.to_records():
        print(f"    lr={row['lr']:.0e}  status={row['status']}")

    # ── 5. UniformSpace — random sampling ──────────────────────────────
    print()
    print("── UniformSpace random sampling ─────────────────────────")
    uniform = UniformSpace({"lr": [1e-3, 5e-4, 1e-4, 5e-5]}, n_samples=3, seed=1)
    rand_exp = ws.project("demo").experiment("random-scan")
    rand_scan = rand_exp.sweep(wf, params=uniform)
    rand_summary = rand_scan.execute()
    for row in rand_summary.to_records():
        print(f"  lr={row['lr']:.0e}  status={row['status']}  loss={row['report']:.4f}")


if __name__ == "__main__":
    main()
