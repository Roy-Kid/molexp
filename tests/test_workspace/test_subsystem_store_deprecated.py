"""Tests for ``Workspace.subsystem_store()`` deprecation warning.

Covers sub-spec ``unify-folder-abstraction-02-workspace-subclassing`` ac-005:

- ``workspace.subsystem_store("agent.sessions")`` emits exactly one
  ``DeprecationWarning`` per call.
- The warning's source location points at the *caller* line, not the
  workspace internals — proving ``stacklevel=2``.
- The returned :class:`SubsystemStore` keeps working: ``.dir()`` returns
  a path under ``<workspace_root>/.subsystems/<kind>/``.

The legacy code path is preserved unchanged by sub-spec 02; sub-spec 03
deletes ``subsystem_store`` entirely.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from molexp.workspace import SubsystemStore, Workspace


def test_subsystem_store_emits_single_deprecation_warning(tmp_path: Path) -> None:
    """One call ⇒ exactly one ``DeprecationWarning``.

    The legacy store is *vended once per kind* (cached on the workspace),
    so a second call with the *same* kind also re-emits the warning —
    each invocation is a separate deprecation event from the caller's
    POV; we only assert the first call's behaviour here.
    """
    ws = Workspace(tmp_path)
    ws.materialize()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        ws.subsystem_store("agent.sessions")

    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep) == 1, f"expected exactly one DeprecationWarning, got {len(dep)}"


def test_subsystem_store_warning_uses_stacklevel_2(tmp_path: Path) -> None:
    """The warning's ``filename`` points at this test file (stacklevel=2).

    ``stacklevel=2`` shifts the warning's reported source up one frame so
    the message names the *caller* of ``subsystem_store()``, not its
    internal implementation. Without ``stacklevel=2`` the filename
    would point at ``workspace.py``; with it, the filename equals
    ``__file__`` (this module).
    """
    ws = Workspace(tmp_path)
    ws.materialize()

    with pytest.warns(DeprecationWarning) as record:
        ws.subsystem_store("agent.sessions")

    first = record.list[0]
    # ``filename`` is the absolute path of the frame the warning
    # was attributed to. With stacklevel=2 it must be this test module.
    assert Path(first.filename) == Path(__file__).resolve(), (
        f"DeprecationWarning attribution should point at the test caller "
        f"(stacklevel=2), but pointed at {first.filename!r} instead."
    )


def test_subsystem_store_return_value_is_usable(tmp_path: Path) -> None:
    """``subsystem_store(kind).dir()`` still returns a valid path.

    The legacy code path stays functional throughout sub-spec 02 —
    only the warning is added. Real consumers (workflow cache, agent
    sessions) keep working while sub-spec 03 prepares the deletion.
    """
    ws = Workspace(tmp_path)
    ws.materialize()

    # Silence the deprecation noise; behaviour is what we care about here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        store = ws.subsystem_store("agent.sessions")

    assert isinstance(store, SubsystemStore)

    sd = store.dir()
    assert sd.is_dir()
    assert sd == tmp_path / ".subsystems" / "agent.sessions"
