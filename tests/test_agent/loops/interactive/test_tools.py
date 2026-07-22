"""Read-only interactive tool set (``molexp.agent.loops.interactive.tools``).

One test per distinct behavior of the three factory-built callables plus the
workspace-confinement boundary (the module's whole reason to exist).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.agent.loops.interactive.tools import readonly_tools


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A small workspace tree: one source file + a top-level README."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def hello():\n    return 'hi there'\n")
    (tmp_path / "README.md").write_text("# Project\nhello world\n")
    return tmp_path


def _tools(root: Path) -> dict[str, object]:
    return {tool.__name__: tool for tool in readonly_tools(root)}


class TestReadonlyTools:
    """``readonly_tools`` — the confined read/list/search callables."""

    def test_exposes_exactly_read_file_list_directory_search_code(self, workspace: Path) -> None:
        assert set(_tools(workspace)) == {"read_file", "list_directory", "search_code"}

    def test_read_file_returns_file_contents(self, workspace: Path) -> None:
        read_file = _tools(workspace)["read_file"]
        assert "hi there" in read_file("src/main.py")  # type: ignore[operator]

    def test_read_file_rejects_parent_traversal(self, workspace: Path) -> None:
        """A ``..`` path is refused before any I/O — the confinement boundary."""
        read_file = _tools(workspace)["read_file"]
        out = read_file("../secret.txt")  # type: ignore[operator]
        assert out.startswith("error:")
        assert ".." in out

    def test_list_directory_lists_entries(self, workspace: Path) -> None:
        listing = _tools(workspace)["list_directory"](".")  # type: ignore[operator]
        assert "src/" in listing
        assert "README.md" in listing

    def test_search_code_finds_matching_line(self, workspace: Path) -> None:
        result = _tools(workspace)["search_code"]("hi there")  # type: ignore[operator]
        assert "src/main.py" in result
