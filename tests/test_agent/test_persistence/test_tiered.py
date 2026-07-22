"""Tests for the generic :class:`TieredResourceStore`.

Pins the agent-layer three-tier resource store (registrations > user >
workspace shadow merge, single-layer reads, malformed-record retention,
lock-serialized + atomic writes). Each test defines its own
``DummySpec`` / ``DummyStore`` subclass so the generic is exercised
without depending on Skill/Tool/MCP.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import pytest

from molexp.agent.persistence.tiered import (
    ResourceSpec,
    Scope,
    TieredResourceStore,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class DummySpec(ResourceSpec):
    """Minimal concrete :class:`ResourceSpec` with the bare-required fields."""

    id: str
    name: str


class DummyStore(TieredResourceStore[DummySpec]):
    """Default test store; ``kind_key`` matches the on-disk wrapper."""

    _registrations: ClassVar[list[DummySpec]] = []


class StoreA(TieredResourceStore[DummySpec]):
    """Subclass A — must not share ``_registrations`` with :class:`StoreB`."""

    _registrations: ClassVar[list[DummySpec]] = []


class StoreB(TieredResourceStore[DummySpec]):
    """Subclass B — independent ``_registrations`` from :class:`StoreA`."""

    _registrations: ClassVar[list[DummySpec]] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(
    spec_id: str,
    *,
    name: str = "default",
    scope: Scope = Scope.USER,
) -> DummySpec:
    """Construct a fully-populated :class:`DummySpec` for register() inputs."""
    now = _now_iso()
    return DummySpec(
        id=spec_id,
        name=name,
        scope=scope,
        shadowed=False,
        valid=True,
        invalid_reason="",
        created_at=now,
        updated_at=now,
    )


def _make_store(tmp_path: Path) -> DummyStore:
    """Construct a fresh :class:`DummyStore` rooted at ``tmp_path``."""
    return DummyStore(
        user_path=tmp_path / "user.json",
        workspace_path=tmp_path / "workspace.json",
        spec_cls=DummySpec,
        kind_key="dummies",
    )


@pytest.fixture(autouse=True)
def _clear_dummy_registrations():
    """Ensure each test starts and ends with empty registrations on doubles."""
    for cls in (DummyStore, StoreA, StoreB):
        cls.clear_registrations()
    yield
    for cls in (DummyStore, StoreA, StoreB):
        cls.clear_registrations()


class TestTieredResourceStore:
    """Behaviors of the generic three-layer resource store."""

    # ── Registration semantics ─────────────────────────────────────────

    def test_register_replaces_entry_with_same_id(self, tmp_path: Path) -> None:
        """Re-registering the same id replaces the prior spec (last write wins)."""
        DummyStore.register(_make_spec("dup", name="v1"))
        DummyStore.register(_make_spec("dup", name="v2"))

        store = _make_store(tmp_path)
        entries = [s for s in store.list_all() if s.id == "dup"]

        assert len(entries) == 1
        assert entries[0].name == "v2"

    def test_registrations_are_isolated_per_subclass(self) -> None:
        """``_registrations`` is per-subclass (``__init_subclass__``), not shared."""
        StoreA.register(_make_spec("only-a", name="alpha"))

        assert [s.id for s in StoreA._registrations] == ["only-a"]
        assert StoreB._registrations == []

    def test_delete_leaves_registrations_untouched(self, tmp_path: Path) -> None:
        """``delete`` never clears code-owned registrations (no file-backed hit)."""
        DummyStore.register(_make_spec("X", name="from-code"))

        store = _make_store(tmp_path)
        deleted = store.delete("X", scope=Scope.USER)

        assert deleted is False
        assert [s.id for s in DummyStore._registrations] == ["X"]

    # ── Shadow / scope resolution ──────────────────────────────────────

    def test_shadow_resolves_workspace_over_user_over_registrations(
        self, tmp_path: Path
    ) -> None:
        """``list_all`` shows all layers; ``get`` resolves workspace > user > reg."""
        DummyStore.register(_make_spec("X", name="from-code"))

        store = _make_store(tmp_path)
        store.create(scope=Scope.USER, id="X", name="from-user")
        store.create(scope=Scope.WORKSPACE, id="X", name="from-workspace")

        all_entries = [s for s in store.list_all() if s.id == "X"]
        assert len(all_entries) == 3

        winner = store.get("X")
        assert winner is not None
        assert winner.name == "from-workspace"
        assert winner.scope == Scope.WORKSPACE
        assert winner.shadowed is False

        losers = [s for s in all_entries if s.scope != Scope.WORKSPACE]
        assert all(loser.shadowed is True for loser in losers)

    def test_get_at_returns_only_named_scope_without_shadow_climb(
        self, tmp_path: Path
    ) -> None:
        """``get_at`` returns the literal layer and ``None`` when that layer is empty."""
        DummyStore.register(_make_spec("X", name="from-code"))

        store = _make_store(tmp_path)
        store.create(scope=Scope.USER, id="X", name="from-user")
        store.create(scope=Scope.WORKSPACE, id="X", name="from-workspace")

        user_hit = store.get_at(Scope.USER, "X")
        assert user_hit is not None and user_hit.name == "from-user"

        ws_hit = store.get_at(Scope.WORKSPACE, "X")
        assert ws_hit is not None and ws_hit.name == "from-workspace"

        empty_store = DummyStore(
            user_path=tmp_path / "empty-user.json",
            workspace_path=tmp_path / "empty-workspace.json",
            spec_cls=DummySpec,
            kind_key="dummies",
        )
        assert empty_store.get_at(Scope.USER, "X") is None
        assert empty_store.get_at(Scope.WORKSPACE, "X") is None

    # ── Validation / robustness ────────────────────────────────────────

    def test_invalid_record_surfaces_with_valid_false(self, tmp_path: Path) -> None:
        """Malformed records surface with ``valid=False``, refuse update, allow delete."""
        user_path = tmp_path / "user.json"
        workspace_path = tmp_path / "workspace.json"

        # Manually craft a malformed record (missing required `name` field).
        user_path.write_text(
            json.dumps(
                {
                    "dummies": {
                        "broken": {
                            "id": "broken",
                            # name intentionally absent
                            "scope": "user",
                            "shadowed": False,
                            "valid": True,
                            "invalid_reason": "",
                            "created_at": _now_iso(),
                            "updated_at": _now_iso(),
                        }
                    }
                }
            )
        )

        store = DummyStore(
            user_path=user_path,
            workspace_path=workspace_path,
            spec_cls=DummySpec,
            kind_key="dummies",
        )
        broken = [s for s in store.list_all() if s.id == "broken"]
        assert len(broken) == 1
        assert broken[0].valid is False
        assert broken[0].invalid_reason != ""

        # `get` returns the invalid entry too.
        fetched = store.get("broken")
        assert fetched is not None and fetched.valid is False

        # `update` refuses (existing record can't be validated for the patch).
        with pytest.raises(ValueError):
            store.update("broken", scope=Scope.USER, name="anything")

        # `delete` succeeds — operators must be able to clean up bad entries.
        assert store.delete("broken", scope=Scope.USER) is True
        assert store.get("broken") is None

    # ── Concurrency + atomicity ────────────────────────────────────────

    def test_concurrent_creates_serialize_without_torn_writes(
        self, tmp_path: Path
    ) -> None:
        """``_lock`` serializes concurrent ``create`` to exactly N entries."""
        store = _make_store(tmp_path)

        n = 8
        barrier = threading.Barrier(n)
        errors: list[BaseException] = []

        def worker(idx: int) -> None:
            try:
                barrier.wait(timeout=5)
                store.create(scope=Scope.WORKSPACE, id=f"id-{idx}", name=f"name-{idx}")
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"worker errors: {errors!r}"
        workspace_entries = store.list_scope(Scope.WORKSPACE)
        assert len(workspace_entries) == n
        assert sorted(s.id for s in workspace_entries) == sorted(
            f"id-{i}" for i in range(n)
        )

    def test_failed_replace_leaves_original_and_no_stale_tmp(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A raising ``os.replace`` preserves the canonical file and cleans up ``.tmp``."""
        store = _make_store(tmp_path)

        # Seed an initial record so we have a baseline file to inspect.
        store.create(scope=Scope.WORKSPACE, id="seed", name="ok")
        workspace_path = tmp_path / "workspace.json"
        original_bytes = workspace_path.read_bytes()

        real_replace = os.replace

        def boom(src, dst, *args, **kwargs):  # noqa: ANN002, ANN003, ANN001
            raise OSError("simulated replace failure")

        monkeypatch.setattr(os, "replace", boom)

        with pytest.raises(OSError):
            store.create(scope=Scope.WORKSPACE, id="should-fail", name="nope")

        # Restore so the test directory state can be inspected normally.
        monkeypatch.setattr(os, "replace", real_replace)

        # Original content survived.
        assert workspace_path.read_bytes() == original_bytes

        # No leftover *.tmp files in the workspace dir.
        leftover = list(tmp_path.glob("*.tmp")) + list(tmp_path.glob("*.tmp.*"))
        assert leftover == [], f"stale tmp files: {leftover!r}"
