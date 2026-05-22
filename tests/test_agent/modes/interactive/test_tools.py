"""Read-only interactive tool set — workspace confinement (ac-004)."""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.agent.modes.interactive.tools import readonly_tools


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A small workspace tree: one source file + a nested package."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def hello():\n    return 'hi there'\n")
    (tmp_path / "README.md").write_text("# Project\nhello world\n")
    return tmp_path


def _tools(root: Path) -> dict[str, object]:
    return {tool.__name__: tool for tool in readonly_tools(root)}


def test_readonly_tools_exposes_exactly_three_named_tools(workspace: Path) -> None:
    tools = _tools(workspace)
    assert set(tools) == {"read_file", "list_directory", "search_code"}


def test_no_write_edit_or_bash_tool(workspace: Path) -> None:
    names = set(_tools(workspace))
    for forbidden in ("write_file", "edit_file", "run_bash", "bash", "shell", "delete"):
        assert forbidden not in names


def test_read_file_returns_contents(workspace: Path) -> None:
    read_file = _tools(workspace)["read_file"]
    assert "hi there" in read_file("src/main.py")  # type: ignore[operator]


def test_read_file_rejects_parent_traversal(workspace: Path) -> None:
    read_file = _tools(workspace)["read_file"]
    with pytest.raises(ValueError, match="\\.\\."):
        read_file("../secret.txt")  # type: ignore[operator]


def test_read_file_rejects_absolute_path_outside_root(workspace: Path) -> None:
    read_file = _tools(workspace)["read_file"]
    with pytest.raises(ValueError):
        read_file("/etc/passwd")  # type: ignore[operator]


def test_read_file_missing_file_raises(workspace: Path) -> None:
    read_file = _tools(workspace)["read_file"]
    with pytest.raises(FileNotFoundError):
        read_file("src/nope.py")  # type: ignore[operator]


def test_list_directory_lists_entries(workspace: Path) -> None:
    list_directory = _tools(workspace)["list_directory"]
    listing = list_directory(".")  # type: ignore[operator]
    assert "src/" in listing
    assert "README.md" in listing


def test_list_directory_rejects_escape(workspace: Path) -> None:
    list_directory = _tools(workspace)["list_directory"]
    with pytest.raises(ValueError):
        list_directory("../..")  # type: ignore[operator]


def test_search_code_finds_pattern(workspace: Path) -> None:
    search_code = _tools(workspace)["search_code"]
    result = search_code("hi there")  # type: ignore[operator]
    assert "src/main.py" in result


def test_search_code_reports_no_match(workspace: Path) -> None:
    search_code = _tools(workspace)["search_code"]
    assert "no matches" in search_code("zzz_nonexistent_token_zzz")  # type: ignore[operator]
