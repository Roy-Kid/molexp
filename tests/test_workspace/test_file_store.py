"""``FileStore`` — the single byte-exit under a root."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from molexp.workspace.file_store import FileStore
from molexp.workspace.fs import FileSystem


class TestFileStore:
    def test_put_writes_text_and_dict(self, tmp_path: Path) -> None:
        store = FileStore(tmp_path)
        text = store.put("a.txt", "hello")
        assert text.read_text() == "hello"
        blob = store.put("n.json", {"n": 1})
        assert '"n"' in blob.read_text()

    def test_put_copies_path_and_skips_samefile(self, tmp_path: Path) -> None:
        store = FileStore(tmp_path)
        src = tmp_path / "src.txt"
        src.write_text("x")
        dest = store.put("copied.txt", src)
        assert dest.read_text() == "x"
        again = store.put("copied.txt", dest)
        assert again.read_text() == "x"

    def test_append_creates_and_appends(self, tmp_path: Path) -> None:
        store = FileStore(tmp_path)
        store.append("logs/run.log", "one")
        store.append("logs/run.log", "two\n")
        assert store.resolve("logs/run.log").read_text() == "one\ntwo\n"

    def test_mkdir_and_resolve(self, tmp_path: Path) -> None:
        store = FileStore(tmp_path)
        d = store.mkdir("work/task")
        assert d.is_dir()
        assert store.resolve("work/task") == d

    def test_put_dict_goes_through_atomic_write_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[Path] = []

        def spy(path: Path, data: object) -> None:
            seen.append(Path(path))
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text('{"ok": true}')

        monkeypatch.setattr("molexp.workspace.file_store.atomic_write_json", spy)
        FileStore(tmp_path).put("run.json", {"a": 1})
        assert seen == [tmp_path / "run.json"]

    def test_put_uses_injected_filesystem(self, tmp_path: Path) -> None:
        class FakeDisk:
            def __init__(self) -> None:
                self.wrote: list[str] = []

            def join(self, *parts: object) -> str:
                return str(Path(*[str(p) for p in parts]))

            def dirname(self, path: object) -> str:
                return str(Path(str(path)).parent)

            def mkdir(self, path: object, *, parents: bool = True, exist_ok: bool = True) -> None:
                Path(str(path)).mkdir(parents=parents, exist_ok=exist_ok)

            def atomic_write_text(
                self, path: object, content: str, *, encoding: str = "utf-8"
            ) -> None:
                Path(str(path)).write_text(content, encoding=encoding)
                self.wrote.append(str(path))

        disk = FakeDisk()
        FileStore(tmp_path, fs=cast(FileSystem, disk)).put("via.txt", "x")
        assert disk.wrote == [str(tmp_path / "via.txt")]
        assert (tmp_path / "via.txt").read_text() == "x"

    def test_rejects_absolute_and_escape(self, tmp_path: Path) -> None:
        store = FileStore(tmp_path)
        with pytest.raises(ValueError, match="relative"):
            store.resolve("/etc/passwd")
        with pytest.raises(ValueError, match="escapes"):
            store.resolve("../outside")
