"""Tests for the molexp asset subsystem."""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.assets import Asset, AssetRepo, register_asset
from molexp.context import RunContext, get_current_context, require_current_context, use_run_context
from molexp.engine import TaskEngine
from molexp.task_base import EmptyConfig, Task


class DummyAssetTask(Task[EmptyConfig, str]):
    def forward(self, *data_args, cfg: EmptyConfig) -> str:
        register_asset("dummy.txt", label="test", meta={"task": self.name})
        return "ok"


def test_asset_as_path_roundtrip() -> None:
    asset = Asset(uri="foo/bar.txt", label="log")
    assert asset.uri == "foo/bar.txt"
    assert asset.label == "log"
    assert str(asset.as_path()) == "foo/bar.txt"


def test_asset_repo_add_replace_list_get_clear() -> None:
    repo = AssetRepo()
    first = repo.add(Asset(uri="file1", label="one"))
    assert repo.list() == [first]

    # replace
    second = repo.add(Asset(uri="file1", label="updated"))
    assets = repo.list()
    assert assets == [second]

    # lookup via str and Path
    assert repo.get_by_uri("file1") is second
    assert repo.get_by_uri(Path("file1")) is second

    repo.clear()
    assert repo.list() == []


def test_context_helpers_behaviour() -> None:
    assert get_current_context() is None
    with pytest.raises(RuntimeError):
        require_current_context()

    ctx = RunContext(asset_repo=AssetRepo(), engine=None, run_id="run-1")
    with use_run_context(ctx):
        assert get_current_context() is ctx

    # context restored to None
    assert get_current_context() is None


def test_register_asset_adds_to_current_context() -> None:
    ctx = RunContext(asset_repo=AssetRepo(), engine=None)
    with use_run_context(ctx):
        asset = register_asset("fake/path.txt", label="log")
        assert ctx.asset_repo.list() == [asset]

    # outside of context should raise
    with pytest.raises(RuntimeError):
        register_asset("another.txt")


def test_engine_integration_records_assets() -> None:
    engine = TaskEngine()
    task = DummyAssetTask()
    assert engine.run(task) == "ok"
    assets = engine.last_run_assets()
    assert len(assets) == 1
    assert assets[0].uri == "dummy.txt"
    assert assets[0].label == "test"
