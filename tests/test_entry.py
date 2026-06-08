"""Tests for entry point registry."""

from pathlib import Path

import pytest

from molexp.entry import _registry, clear_registry, entry, load_workspaces
from molexp.workspace import Workspace
from molexp.workspace.workspace import set_cli_root_override


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


@pytest.fixture
def restore_cli_root_override():
    """Guarantee the module-global override is cleared after the test.

    The override is process-global state in ``molexp.workspace.workspace``;
    leaking a non-``None`` value would silently rewrite the root of every
    ``Workspace(...)`` constructed in later tests.
    """
    set_cli_root_override(None)
    try:
        yield
    finally:
        set_cli_root_override(None)


class TestInferWorkspaceRoot:
    """ac-001 / ac-002 — the pure path helper."""

    def test_returns_script_parent_directory(self, tmp_path):
        # ac-001: infer_workspace_root(.../script.py) == resolved parent dir.
        from molexp.entry import infer_workspace_root

        script = tmp_path / "sub" / "script.py"
        script.parent.mkdir(parents=True)
        script.write_text("x = 1\n")

        assert infer_workspace_root(script) == script.resolve().parent
        assert infer_workspace_root(script) == (tmp_path / "sub").resolve()

    def test_matches_literal_parent_for_absolute_path(self):
        # ac-001: a/b/script.py -> a/b (resolved).
        from molexp.entry import infer_workspace_root

        assert infer_workspace_root(Path("/a/b/script.py")) == Path("/a/b").resolve()

    def test_empty_path_raises_value_error(self):
        # ac-002: fail fast on a falsy / unresolvable path, no silent default.
        from molexp.entry import infer_workspace_root

        with pytest.raises(ValueError):
            infer_workspace_root(Path())


class TestWorkspaceRootInference:
    """ac-003 / ac-004 / ac-005 — the constructor boundary."""

    def test_rootless_workspace_uses_cli_override(self, tmp_path, restore_cli_root_override):
        # ac-003: Workspace(name=...) with no root resolves to the override.
        override_dir = tmp_path / "override"
        override_dir.mkdir()
        set_cli_root_override(override_dir)

        ws = Workspace(name="x")

        assert ws.root == override_dir.resolve()

    def test_rootless_workspace_without_override_raises(self, restore_cli_root_override):
        # ac-004: no root and no active override -> clear ValueError.
        with pytest.raises(ValueError):
            Workspace(name="x")

    def test_explicit_root_unaffected_by_inference(self, tmp_path, restore_cli_root_override):
        # ac-005: explicit root resolves to that path when no override is set.
        explicit = tmp_path / "explicit"

        ws = Workspace(explicit, name="x")

        assert ws.root == explicit.resolve()
