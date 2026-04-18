"""Tests for entry point registry."""

import pytest

from molexp.entry import _registry, clear_registry, entry, load_workspaces
from molexp.workspace import Workspace


@pytest.fixture(autouse=True)
def clean_registry():
    clear_registry()
    yield
    clear_registry()


class TestEntry:
    def test_entry_populates_registry(self, tmp_path):
        ws = Workspace(tmp_path / "ws", name="ws")
        entry(ws)
        assert len(_registry) == 1
        assert _registry[0] is ws

    def test_multiple_entries(self, tmp_path):
        a = Workspace(tmp_path / "a", name="a")
        b = Workspace(tmp_path / "b", name="b")
        entry(a)
        entry(b)
        assert len(_registry) == 2

    def test_clear_registry(self, tmp_path):
        entry(Workspace(tmp_path / "ws", name="ws"))
        clear_registry()
        assert len(_registry) == 0


class TestLoadWorkspaces:
    def test_load_from_script(self, tmp_path):
        ws_path = tmp_path / "ws"
        script = tmp_path / "test_script.py"
        script.write_text(
            "from molexp.workspace import Workspace\n"
            "from molexp.entry import entry\n"
            f"ws = Workspace({str(ws_path)!r}, name='from-script')\n"
            "entry(ws)\n"
        )
        workspaces = load_workspaces(script)
        assert len(workspaces) == 1
        assert workspaces[0].name == "from-script"

    def test_load_clears_previous_entries(self, tmp_path):
        entry(Workspace(tmp_path / "stale", name="stale"))

        ws_path = tmp_path / "fresh"
        script = tmp_path / "test_script.py"
        script.write_text(
            "from molexp.workspace import Workspace\n"
            "from molexp.entry import entry\n"
            f"entry(Workspace({str(ws_path)!r}, name='fresh'))\n"
        )
        workspaces = load_workspaces(script)
        assert len(workspaces) == 1
        assert workspaces[0].name == "fresh"

    def test_load_empty_script(self, tmp_path):
        script = tmp_path / "empty.py"
        script.write_text("x = 1\n")
        workspaces = load_workspaces(script)
        assert workspaces == []

    def test_load_invalid_script(self, tmp_path):
        script = tmp_path / "bad.py"
        script.write_text("raise ImportError('boom')\n")
        with pytest.raises(ImportError):
            load_workspaces(script)
