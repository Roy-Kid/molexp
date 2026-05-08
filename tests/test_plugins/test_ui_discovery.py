"""Tests for ``molexp.plugins.ui.discover_ui_plugin_dirs`` — the slim,
Python-side directory-pointer discovery for UI plugins.

This module deliberately has zero UI semantics on the Python side: the
discovery only resolves a directory pointer per entry-point. The real
manifest (api version, contributions, etc.) lives in TypeScript land.

These tests are isolated from the host process's installed entry points
by monkeypatching ``importlib.metadata.entry_points`` so the suite stays
deterministic regardless of which optional plugins the developer has
installed locally.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest

from molexp.plugins.ui import _discover_ui_uncached, discover_ui_plugin_dirs

# ── fakes / fixtures ──────────────────────────────────────────────────────


class _FakeEntryPoint:
    """Minimal ``importlib.metadata.EntryPoint`` substitute."""

    def __init__(
        self,
        name: str,
        loader,
        *,
        group: str = "molexp.ui_plugins",
    ) -> None:
        self.name = name
        self.group = group
        self._loader = loader

    def load(self):
        return self._loader()


def _install_fake_eps(
    monkeypatch: pytest.MonkeyPatch,
    eps: Iterable[_FakeEntryPoint],
) -> None:
    eps_tuple = tuple(eps)

    class _FakeEntryPoints:
        def select(self, *, group: str):
            return tuple(ep for ep in eps_tuple if ep.group == group)

    monkeypatch.setattr(
        "molexp.plugins.ui.importlib_metadata.entry_points",
        lambda: _FakeEntryPoints(),
    )
    # Cached state must be cleared so the test sees the patched eps
    _discover_ui_uncached.cache_clear()


@pytest.fixture
def warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Spy on ``molexp.plugins.ui.logger.warning`` calls.

    mollog bypasses stdlib ``logging`` so pytest's ``caplog`` / ``capfd``
    do not see its output; we capture the messages directly.
    """
    captured: list[str] = []
    import molexp.plugins.ui as ui_mod

    monkeypatch.setattr(
        ui_mod.logger,
        "warning",
        lambda msg, **_: captured.append(str(msg)),
    )
    return captured


# ── discovery tests (ac-003) ──────────────────────────────────────────────


class TestDiscoverUiPluginDirs:
    def test_returns_dict_keyed_by_ep_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_fake_eps(
            monkeypatch,
            [_FakeEntryPoint("alpha", lambda: tmp_path)],
        )

        result = discover_ui_plugin_dirs()

        assert isinstance(result, dict)
        assert result == {"alpha": tmp_path}

    def test_path_loader_accepted(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        # Loader returns a Path directly (not a callable wrapping one).
        _install_fake_eps(
            monkeypatch,
            [_FakeEntryPoint("direct", lambda: tmp_path)],
        )

        result = discover_ui_plugin_dirs()

        assert result == {"direct": tmp_path}

    def test_callable_loader_accepted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Loader returns a zero-arg callable that returns a Path.
        def _resolve() -> Path:
            return tmp_path

        _install_fake_eps(
            monkeypatch,
            [_FakeEntryPoint("callable", lambda: _resolve)],
        )

        result = discover_ui_plugin_dirs()

        assert result == {"callable": tmp_path}

    def test_callable_raising_is_isolated(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        warnings: list[str],
    ) -> None:
        good_dir = tmp_path / "good"
        good_dir.mkdir()

        def _explode() -> Path:
            raise RuntimeError("boom")

        _install_fake_eps(
            monkeypatch,
            [
                _FakeEntryPoint("bad", lambda: _explode),
                _FakeEntryPoint("good", lambda: good_dir),
            ],
        )

        result = discover_ui_plugin_dirs()

        assert result == {"good": good_dir}
        assert any("bad" in msg for msg in warnings)

    def test_non_directory_path_is_filtered(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        warnings: list[str],
    ) -> None:
        missing_path = tmp_path / "nope"  # does not exist
        file_path = tmp_path / "file.txt"
        file_path.touch()  # exists but is a file, not a directory

        _install_fake_eps(
            monkeypatch,
            [
                _FakeEntryPoint("missing", lambda: missing_path),
                _FakeEntryPoint("isfile", lambda: file_path),
            ],
        )

        result = discover_ui_plugin_dirs()

        assert result == {}
        assert any("missing" in msg for msg in warnings)
        assert any("isfile" in msg for msg in warnings)

    def test_duplicate_id_first_wins(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        warnings: list[str],
    ) -> None:
        first_dir = tmp_path / "first"
        first_dir.mkdir()
        second_dir = tmp_path / "second"
        second_dir.mkdir()

        _install_fake_eps(
            monkeypatch,
            [
                _FakeEntryPoint("dup", lambda: first_dir),
                _FakeEntryPoint("dup", lambda: second_dir),
            ],
        )

        result = discover_ui_plugin_dirs()

        assert result == {"dup": first_dir}
        assert any("duplicate" in msg.lower() for msg in warnings)

    def test_no_entry_points_returns_empty_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_eps(monkeypatch, [])

        assert discover_ui_plugin_dirs() == {}

    def test_entry_point_load_failure_is_isolated(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        warnings: list[str],
    ) -> None:
        good_dir = tmp_path / "good"
        good_dir.mkdir()

        def _import_boom():
            raise ImportError("missing dep")

        _install_fake_eps(
            monkeypatch,
            [
                _FakeEntryPoint("bad", _import_boom),
                _FakeEntryPoint("good", lambda: good_dir),
            ],
        )

        result = discover_ui_plugin_dirs()

        assert result == {"good": good_dir}
        assert any("bad" in msg for msg in warnings)

    def test_cache_hits_on_second_call(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Wrap the fake entry_points() in a counter so we can assert that
        # the second call to ``discover_ui_plugin_dirs`` does NOT
        # re-enumerate (i.e. ``functools.cache`` works).
        eps_tuple = (_FakeEntryPoint("cached", lambda: tmp_path),)
        call_count = {"n": 0}

        class _FakeEntryPoints:
            def select(self, *, group: str):
                return tuple(ep for ep in eps_tuple if ep.group == group)

        def _entry_points():
            call_count["n"] += 1
            return _FakeEntryPoints()

        monkeypatch.setattr(
            "molexp.plugins.ui.importlib_metadata.entry_points",
            _entry_points,
        )
        # Reset cache only at the start; do not reset between the two calls.
        _discover_ui_uncached.cache_clear()

        first = discover_ui_plugin_dirs()
        second = discover_ui_plugin_dirs()

        assert first == {"cached": tmp_path}
        assert second == first
        assert call_count["n"] == 1

    def test_module_does_not_define_uiplugin_class(self) -> None:
        # Sanity check for the design: Python side has zero UI semantics —
        # no ``UiPlugin`` dataclass, no ``api_version`` field. UI semantics
        # live in TS-side ``manifest.json``.
        import molexp.plugins.ui as ui_mod

        assert not hasattr(ui_mod, "UiPlugin")
