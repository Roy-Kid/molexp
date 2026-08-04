"""RemoteFileSystem.join must preserve absolute remote roots."""

from __future__ import annotations

from molexp.workspace.fs_remote import RemoteFileSystem


class TestRemoteFileSystemJoin:
    def test_join_preserves_absolute_root(self) -> None:
        assert (
            RemoteFileSystem.join("/home/jicli594/work/pinet-quant-raw", "workspace.json")
            == "/home/jicli594/work/pinet-quant-raw/workspace.json"
        )

    def test_join_relative_parts(self) -> None:
        assert (
            RemoteFileSystem.join("projects", "water", "project.json")
            == "projects/water/project.json"
        )

    def test_join_empty_parts_skipped(self) -> None:
        assert RemoteFileSystem.join("/root", "", "a") == "/root/a"

    def test_join_no_parts(self) -> None:
        assert RemoteFileSystem.join() == ""

    def test_is_absolute_still_detects_root(self) -> None:
        assert RemoteFileSystem.is_absolute("/home/x")
        assert not RemoteFileSystem.is_absolute("home/x")
