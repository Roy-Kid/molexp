"""Unit tests for :class:`molexp.workflow.materialization_store.FileMaterializationStore`.

The engine-side materialization store owns two responsibilities: a deterministic
content-addressed working directory (``workdir_for``) and fail-soft return-value
persistence with retrievable lineage (``persist_result``).
"""

from __future__ import annotations

from pathlib import Path

from molexp.workflow.materialization_store import FileMaterializationStore
from molexp.workspace import Workspace


def _new_run(tmp_path: Path):
    ws = Workspace(tmp_path / "lab")
    project = ws.add_project(name="p")
    experiment = project.add_experiment(name="e")
    return experiment.add_run(params={})


class TestFileMaterializationStore:
    def test_workdir_for_is_deterministic(self, tmp_path: Path) -> None:
        store = FileMaterializationStore(tmp_path / "mat")
        assert store.workdir_for("sha256:deadbeefcafe") == store.workdir_for("sha256:deadbeefcafe")

    def test_workdir_for_keeps_both_terms_of_code_config_key(self, tmp_path: Path) -> None:
        # A "code:config" key must not collapse to just the config term.
        store = FileMaterializationStore(tmp_path / "mat")
        assert store.workdir_for("codeA:cfg") != store.workdir_for("codeB:cfg")

    def test_persist_result_returns_sha256_and_registers_lineage(self, tmp_path: Path) -> None:
        store = FileMaterializationStore(tmp_path / "mat")
        run = _new_run(tmp_path)
        with run.start() as ctx:
            ctx.set_active_task("produce")
            content_hash = store.persist_result("produce", {"value": 42}, run_context=ctx)
        assert content_hash is not None
        assert content_hash.startswith("sha256:")
        found = run.assets.query(producer_task="produce", kind="artifact")
        assert any(getattr(a, "content_hash", None) == content_hash for a in found)

    def test_persist_result_none_run_context_returns_none(self, tmp_path: Path) -> None:
        store = FileMaterializationStore(tmp_path / "mat")
        assert store.persist_result("t", {"a": 1}, run_context=None) is None

    def test_persist_result_non_json_safe_returns_none(self, tmp_path: Path) -> None:
        store = FileMaterializationStore(tmp_path / "mat")
        run = _new_run(tmp_path)

        class _Unserializable:
            pass

        with run.start() as ctx:
            ctx.set_active_task("produce")
            # An object json.dumps cannot encode → fail-soft, no raise, returns None.
            result = store.persist_result("produce", {"obj": _Unserializable()}, run_context=ctx)
        assert result is None
