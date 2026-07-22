"""Tests for :class:`molexp.workspace.gitignore.GitIgnoreMatcher`."""

from __future__ import annotations

from pathlib import Path

from molexp.workspace.gitignore import GitIgnoreMatcher, load_gitignore_matcher


class TestGitIgnoreMatcher:
    def test_safety_floor_ignores_vcs_and_caches_without_gitignore(self, tmp_path: Path) -> None:
        """The always-on safety floor applies even with no on-disk ``.gitignore``."""
        root = tmp_path / "ws"
        root.mkdir()
        m = load_gitignore_matcher(root)
        assert m.is_ignored("node_modules", is_dir=True)
        assert m.is_ignored("node_modules/pkg/index.js", is_dir=False)
        assert m.is_ignored(".git", is_dir=True)
        assert m.is_ignored(".venv", is_dir=True)
        assert not m.is_ignored("projects", is_dir=True)
        assert not m.is_ignored("notes/alpha.md", is_dir=False)

    def test_root_gitignore_patterns_and_negation(self, tmp_path: Path) -> None:
        root = tmp_path / "ws"
        root.mkdir()
        (root / ".gitignore").write_text(
            "\n".join(
                [
                    "scratch/",
                    "*.log",
                    "!important.log",
                    "build",
                ]
            )
            + "\n"
        )
        m = load_gitignore_matcher(root)
        assert m.is_ignored("scratch", is_dir=True)
        assert m.is_ignored("scratch/tmp.txt", is_dir=False)
        assert m.is_ignored("run.log", is_dir=False)
        assert not m.is_ignored("important.log", is_dir=False)
        assert m.is_ignored("build", is_dir=True)
        assert m.is_ignored("build/out.bin", is_dir=False)
        assert not m.is_ignored("src/main.py", is_dir=False)

    def test_nested_gitignore_cascade_and_scoped_negation(self, tmp_path: Path) -> None:
        root = tmp_path / "ws"
        (root / "data").mkdir(parents=True)
        (root / ".gitignore").write_text("*.tmp\n")
        (root / "data" / ".gitignore").write_text("secret/\n!keep.tmp\n")

        m = load_gitignore_matcher(root)
        assert m.is_ignored("foo.tmp", is_dir=False)
        assert m.is_ignored("data/secret", is_dir=True)
        assert m.is_ignored("data/secret/x", is_dir=False)
        # Nested negation only un-ignores within that file's scope: keep.tmp under
        # data/ was ignored by root *.tmp, nested !keep.tmp re-includes it.
        assert not m.is_ignored("data/keep.tmp", is_dir=False)
        # Root-level keep.tmp has no nested rule that reaches it.
        assert m.is_ignored("keep.tmp", is_dir=False)

    def test_root_path_is_never_ignored(self, tmp_path: Path) -> None:
        """Even under a catch-all ``*`` the root itself must never be skipped."""
        root = tmp_path / "ws"
        root.mkdir()
        (root / ".gitignore").write_text("*\n")
        m = GitIgnoreMatcher(root)
        assert not m.is_ignored("", is_dir=True)
        assert not m.is_ignored("/", is_dir=True)
