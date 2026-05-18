"""Failing tests pinning the contract for ``molexp.plugins.cli``.

These tests cover the new CLI-only plugin layer introduced by
spec ``07-cli-ui-plugin-split``:

- a frozen ``CliPlugin`` dataclass with a *required* ``register``
  callable (no more ``register_cli: Callable | None``)
- a ``CLI_PLUGIN_API_VERSION`` constant pinned to ``"1"``
- a ``discover_cli_plugins()`` function that walks the
  ``molexp.cli_plugins`` entry-point group with
  ``functools.cache`` + failure-isolation + first-wins
  de-duplication

The tests are deliberately RED right now: ``molexp.plugins.cli`` does
not exist yet, so the very first import fails with ``ImportError``.
Implementation is scheduled for Step 5 of the spec.
"""

from __future__ import annotations

from collections.abc import Iterable

import pytest
import typer

from molexp.plugins.cli import (  # type: ignore[import-not-found]
    CLI_PLUGIN_API_VERSION,
    CliPlugin,
    _discover_cli_uncached,  # type: ignore[import-not-found]
    discover_cli_plugins,
)

# ── contract dataclass tests (ac-001) ─────────────────────────────────────


class TestCliPluginContract:
    def test_is_immutable_plain_class(self) -> None:
        """CliPlugin is a plain Python class with explicit __init__; it carries
        a live ``register`` callable so it is not a pydantic BaseModel."""
        from pydantic import BaseModel

        assert not issubclass(CliPlugin, BaseModel)

    def test_required_fields(self) -> None:
        def reg(app: typer.Typer) -> None:
            pass

        plugin = CliPlugin(id="x", name="X", version="0.0.1", register=reg)
        assert plugin.id == "x"
        assert plugin.name == "X"
        assert plugin.version == "0.0.1"
        assert plugin.api_version == "1"
        assert callable(plugin.register)
        assert plugin.register is reg

        # ``register`` is a required, non-defaulted field — constructing
        # a CliPlugin without it must raise ``TypeError``.
        with pytest.raises(TypeError):
            CliPlugin(id="x", name="X", version="0.0.1")  # type: ignore[call-arg]

    def test_immutable(self) -> None:
        plugin = CliPlugin(
            id="x",
            name="X",
            version="0.0.1",
            register=lambda app: None,  # noqa: ARG005
        )
        with pytest.raises(AttributeError):
            plugin.id = "y"  # type: ignore[misc]

    def test_api_version_constant(self) -> None:
        assert CLI_PLUGIN_API_VERSION == "1"
        plugin = CliPlugin(
            id="x",
            name="X",
            version="0.0.1",
            register=lambda app: None,  # noqa: ARG005
        )
        assert plugin.api_version == CLI_PLUGIN_API_VERSION


# ── discovery tests (ac-002) ──────────────────────────────────────────────


class _FakeEntryPoint:
    """Minimal ``importlib.metadata.EntryPoint`` substitute."""

    def __init__(
        self,
        name: str,
        loader,
        *,
        group: str = "molexp.cli_plugins",
    ) -> None:
        self.name = name
        self.group = group
        self._loader = loader

    def load(self):
        return self._loader()


def _install_fake_eps(
    monkeypatch: pytest.MonkeyPatch,
    eps: Iterable[_FakeEntryPoint],
) -> list[int]:
    """Install fake entry points and return a call-counter list.

    The returned ``list[int]`` has length 1 and tracks how many times
    the patched ``entry_points()`` callable is invoked — used by the
    cache test to assert the second call hits the cache.
    """
    eps_tuple = tuple(eps)
    call_count = [0]

    class _FakeEntryPoints:
        def select(self, *, group: str):
            return tuple(ep for ep in eps_tuple if ep.group == group)

    def _entry_points():
        call_count[0] += 1
        return _FakeEntryPoints()

    monkeypatch.setattr(
        "molexp.plugins.cli.importlib_metadata.entry_points",
        _entry_points,
    )
    # Cached state must be cleared so the test sees the patched eps
    _discover_cli_uncached.cache_clear()
    return call_count


@pytest.fixture
def warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Spy on ``molexp.plugins.cli.logger.warning`` calls.

    mollog bypasses stdlib ``logging`` so pytest's ``caplog`` / ``capfd``
    do not see its output; we capture the messages directly.
    """
    captured: list[str] = []
    from molexp.plugins import cli as cli_mod  # type: ignore[import-not-found]

    monkeypatch.setattr(
        cli_mod.logger,
        "warning",
        lambda msg, **_: captured.append(str(msg)),
    )
    return captured


def _make_plugin(
    plugin_id: str = "alpha",
    *,
    api_version: str = CLI_PLUGIN_API_VERSION,
) -> CliPlugin:
    return CliPlugin(
        id=plugin_id,
        name=plugin_id.title(),
        version="1.0.0",
        register=lambda app: None,  # noqa: ARG005
        api_version=api_version,
    )


class TestDiscoverCliPlugins:
    def test_happy_path_returns_tuple(self, monkeypatch: pytest.MonkeyPatch) -> None:
        valid = _make_plugin("alpha")
        _install_fake_eps(monkeypatch, [_FakeEntryPoint("alpha", lambda: valid)])

        result = discover_cli_plugins()

        assert isinstance(result, tuple)
        assert result == (valid,)

    def test_failing_entry_point_is_isolated(
        self,
        monkeypatch: pytest.MonkeyPatch,
        warnings: list[str],
    ) -> None:
        valid = _make_plugin("good")

        def boom():
            raise ImportError("missing dep")

        _install_fake_eps(
            monkeypatch,
            [
                _FakeEntryPoint("bad", boom),
                _FakeEntryPoint("good", lambda: valid),
            ],
        )

        result = discover_cli_plugins()

        assert result == (valid,)
        assert any("bad" in msg for msg in warnings)

    def test_non_cliplugin_object_is_filtered(
        self,
        monkeypatch: pytest.MonkeyPatch,
        warnings: list[str],
    ) -> None:
        _install_fake_eps(
            monkeypatch,
            [_FakeEntryPoint("not-a-plugin", lambda: object())],
        )

        result = discover_cli_plugins()

        assert result == ()
        assert any("CliPlugin" in msg for msg in warnings)

    def test_wrong_api_version_is_filtered(
        self,
        monkeypatch: pytest.MonkeyPatch,
        warnings: list[str],
    ) -> None:
        future = _make_plugin("future", api_version="999")
        valid = _make_plugin("now")
        _install_fake_eps(
            monkeypatch,
            [
                _FakeEntryPoint("future", lambda: future),
                _FakeEntryPoint("now", lambda: valid),
            ],
        )

        result = discover_cli_plugins()

        assert result == (valid,)
        assert any("api_version" in msg for msg in warnings)

    def test_duplicate_id_first_wins(
        self,
        monkeypatch: pytest.MonkeyPatch,
        warnings: list[str],
    ) -> None:
        first = _make_plugin("dup")
        second = _make_plugin("dup")
        assert first is not second  # sanity: distinct objects, same id

        _install_fake_eps(
            monkeypatch,
            [
                _FakeEntryPoint("first", lambda: first),
                _FakeEntryPoint("second", lambda: second),
            ],
        )

        result = discover_cli_plugins()

        assert result == (first,)
        assert any("duplicate" in msg.lower() for msg in warnings)

    def test_no_entry_points_returns_empty_tuple(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_eps(monkeypatch, [])
        assert discover_cli_plugins() == ()

    def test_cache_hits_on_second_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        valid = _make_plugin("cached")
        call_count = _install_fake_eps(
            monkeypatch,
            [_FakeEntryPoint("cached", lambda: valid)],
        )

        first = discover_cli_plugins()
        calls_after_first = call_count[0]
        second = discover_cli_plugins()

        assert first == second == (valid,)
        # entry_points() must NOT be called again on the cached path
        assert call_count[0] == calls_after_first
        assert calls_after_first == 1
