"""Log append write-amplification (workflow-workspace-hardening P1-7).

``ctx.log(name).append(line)`` used to rewrite the *entire* scope manifest
(``assets.json``) AND write a catalog row on **every line** — O(lines x assets)
disk churn. The line bytes still go straight to the ``.log`` file (an O(1)
append), but the ``line_count`` / ``updated_at`` metadata is now deferred and
flushed once when the run context exits. Appending K lines therefore performs
zero manifest rewrites and zero catalog writes during the loop; the final
metadata still lands on exit.
"""

from __future__ import annotations

import pytest

from molexp.workspace import Workspace
from molexp.workspace.assets.manifest import AssetManifest
from molexp.workspace.catalog.index import AssetCatalog


@pytest.fixture
def run(tmp_path):
    ws = Workspace(root=tmp_path / "lab", name="lab")
    exp = ws.add_project("p").add_experiment("e")
    return exp.add_run(params={"i": 0})


def test_append_does_not_rewrite_manifest_or_catalog_per_line(run, monkeypatch):
    saves = {"manifest": 0, "catalog": 0}

    orig_save = AssetManifest._save_raw
    orig_cat_update = AssetCatalog.update

    def counting_save(self, assets):
        saves["manifest"] += 1
        return orig_save(self, assets)

    def counting_cat_update(self, asset):
        saves["catalog"] += 1
        return orig_cat_update(self, asset)

    monkeypatch.setattr(AssetManifest, "_save_raw", counting_save)
    monkeypatch.setattr(AssetCatalog, "update", counting_cat_update)

    with run.start() as ctx:
        log = ctx.log("train")  # creation may register (one-time), that's fine
        saves["manifest"] = 0
        saves["catalog"] = 0
        for i in range(50):
            log.append(f"line {i}")
        # The 50 appends must not rewrite the manifest or touch the catalog.
        assert saves["manifest"] == 0, f"manifest rewritten {saves['manifest']}x during appends"
        assert saves["catalog"] == 0, f"catalog written {saves['catalog']}x during appends"


def test_metadata_flushed_on_exit_and_content_intact(run):
    with run.start() as ctx:
        log = ctx.log("train")
        for i in range(10):
            log.append(f"line {i}")
        # tail reads the .log file directly — content is live regardless of flush.
        assert log.tail() == [f"line {i}" for i in range(10)]

    # After exit, the manifest reflects the final line_count.
    catalog = run.experiment.project.workspace.catalog
    logs = [a for a in catalog.query_assets(kind="log", producer_run=run.id) if a.name == "train"]
    assert len(logs) == 1
    assert logs[0].line_count == 10
