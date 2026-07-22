"""``molexp.workspace.git_projection`` — shared CLI≡server backend + the push verb.

Spec: workspace-git-projection-04-wire. The ``molexp git`` CLI and the
``/api/git/*`` server routes call the SAME backend symbols (Python ≡ UI). The
``push`` backend mirrors the projected ``refs/molexp/*`` to a remote.

Whether ``git_push`` is *classified* destructive (owned by
``test_harness/test_curation_capability_catalog.py``) and how a destructive
capability is *gated* (owned by ``test_harness/test_side_effect_gate.py``) are
harness concerns and are not re-tested here.
"""

from __future__ import annotations

import subprocess

import molexp.workspace.git_projection as gp
from molexp.workspace import Workspace


class TestSharedBackend:
    """Python ≡ UI: CLI and server bind the identical backend functions."""

    def test_cli_and_server_reference_the_same_backend(self) -> None:
        from molexp.cli import git_cmd as cli_git
        from molexp.server.routes import git as route_git

        assert cli_git.checkpoint is gp.checkpoint
        assert route_git.checkpoint is gp.checkpoint
        assert cli_git.rebuild is gp.rebuild
        assert route_git.rebuild is gp.rebuild
        assert cli_git.push is gp.push
        assert route_git.push is gp.push


class TestPush:
    """``push`` mirrors the locally-materialized refs to a remote."""

    async def test_push_mirrors_projected_refs_to_remote(self, tmp_path) -> None:
        ws = Workspace(root=tmp_path / "lab", name="Lab")
        run = ws.add_project("demo").add_experiment("baseline", params={}).add_run(params={})
        with run.start() as ctx:
            ctx.artifact.save("m.json", {"v": 1})
        await gp.checkpoint(ws)  # local materialization builds the refs

        remote = tmp_path / "remote.git"
        subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
        await gp.push(ws, remote=str(remote))

        shown = subprocess.run(
            ["git", "-C", str(remote), "show-ref"], capture_output=True, text=True
        ).stdout
        assert f"refs/molexp/runs/{run.id}" in shown
