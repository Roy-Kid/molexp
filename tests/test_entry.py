"""Tests for entry point registry."""

import pytest
from pathlib import Path
import tempfile

from molexp.entry import entry, clear_registry, load_projects, _registry
from molexp.project import Project


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure registry is clean before and after each test."""
    clear_registry()
    yield
    clear_registry()


class TestEntry:
    def test_entry_populates_registry(self):
        p = Project("test")
        entry(p)
        assert len(_registry) == 1
        assert _registry[0] is p

    def test_multiple_entries(self):
        p1 = Project("a")
        p2 = Project("b")
        entry(p1)
        entry(p2)
        assert len(_registry) == 2

    def test_clear_registry(self):
        entry(Project("test"))
        clear_registry()
        assert len(_registry) == 0


class TestLoadProjects:
    def test_load_from_script(self, tmp_path):
        script = tmp_path / "test_script.py"
        script.write_text(
            "from molexp.project import Project\n"
            "from molexp.entry import entry\n"
            "p = Project('from-script')\n"
            "entry(p)\n"
        )
        projects = load_projects(script)
        assert len(projects) == 1
        assert projects[0].name == "from-script"

    def test_load_clears_previous_entries(self, tmp_path):
        entry(Project("stale"))

        script = tmp_path / "test_script.py"
        script.write_text(
            "from molexp.project import Project\n"
            "from molexp.entry import entry\n"
            "entry(Project('fresh'))\n"
        )
        projects = load_projects(script)
        assert len(projects) == 1
        assert projects[0].name == "fresh"

    def test_load_empty_script(self, tmp_path):
        script = tmp_path / "empty.py"
        script.write_text("x = 1\n")
        projects = load_projects(script)
        assert projects == []

    def test_load_invalid_script(self, tmp_path):
        script = tmp_path / "bad.py"
        script.write_text("raise ImportError('boom')\n")
        with pytest.raises(ImportError):
            load_projects(script)
