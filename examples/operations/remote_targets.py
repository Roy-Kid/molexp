"""Compute targets — LocalTarget, RemoteTarget, and target resolution.

Matches ``docs/guide/workspace-architecture.md``.

Demonstrates:

1. ``LocalTarget`` — execution on the local filesystem.
2. ``RemoteTarget`` — execution on a remote host via SSH.
3. ``ComputeTarget`` — the persisted base (stored in workspace config).
4. ``resolve_compute_target(ws, name)`` — named-target resolution.
5. Target registration in workspace configuration.

Run directly::

    python examples/operations/remote_targets.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import molexp as me
from molexp.workspace.target import LocalTarget, RemoteTarget
from molexp.workspace.targets import resolve_compute_target


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="molexp-targets-"))
    ws = me.Workspace(root, name="targets-demo")

    scratch = Path(tempfile.mkdtemp(prefix="molexp-scratch-"))

    # ── 1. LocalTarget — the default for in-process execution ────────────
    local = LocalTarget(scratch_root=str(scratch))
    print(f"LocalTarget:    {local}")
    print(f"  is_remote:    {local.is_remote}")
    print(f"  scratch_root: {local.scratch_root}")

    # ── 2. RemoteTarget — SSH reachable host ─────────────────────────────
    remote = RemoteTarget(host="login.hpc.example.com", user="alice", scratch_root="/scratch/alice")
    print(f"RemoteTarget:   {remote}")
    print(f"  host:         {remote.host}")
    print(f"  user:         {remote.user}")
    print(f"  is_remote:    {remote.is_remote}")

    # ── 3. ComputeTarget — the persisted base stored in workspace.json ───
    from molexp.workspace.target import ComputeTarget

    ct = ComputeTarget(name="gpu-cluster", host="gpu01.hpc.example.com", scratch_root="/scratch")
    print(f"\nComputeTarget(name='{ct.name}'):")
    print(f"  host:      {ct.host}")
    print(f"  is_remote: {ct.is_remote}")

    # ── 4. resolve_compute_target — resolve built-in "local" ────────────
    resolved = resolve_compute_target(ws, "local")
    print("\nresolve_compute_target(ws, 'local'):")
    print(f"  name:      {resolved.name}")
    print(f"  is_remote: {resolved.is_remote}")

    # ── 5. Register a target in workspace config ─────────────────────────
    # ComputeTargets are persisted in the workspace config; resolve by name
    # after adding them through the workspace's target API.
    print("\nTarget resolution ready — construct ComputeTarget, then resolve by name.")


if __name__ == "__main__":
    main()
