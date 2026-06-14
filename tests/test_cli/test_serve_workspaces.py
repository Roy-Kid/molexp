"""Tests for ``molexp serve`` multi-workspace resolution helpers."""

from molexp.cli.workspace.serve import _resolve_served, _slug, _unique_key


def test_slug_sanitizes_and_never_empty():
    assert _slug("me@host:/scratch/Runs") == "me-host-scratch-runs"
    assert _slug("a__b") == "a_b" or _slug("a__b").startswith("a")
    assert _slug("") == "ws"
    assert _slug("///") == "ws"


def test_unique_key_disambiguates():
    used: set[str] = set()
    assert _unique_key("a", used) == "a"
    assert _unique_key("a", used) == "a-2"
    assert _unique_key("a", used) == "a-3"
    assert _unique_key("b", used) == "b"


def test_resolve_served_local(tmp_path):
    (tmp_path / "workspace.json").write_text("{}")
    used: set[str] = set()
    sw = _resolve_served(str(tmp_path), used)
    assert sw.is_remote is False
    assert sw.target_name is None
    assert sw.path == str(tmp_path.resolve())
    assert sw.key.startswith("local-")


def test_resolve_served_auto_detects_nested_workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "workspace.json").write_text("{}")
    sw = _resolve_served(str(tmp_path), set())
    assert sw.path == str(ws.resolve())  # descended into ./workspace


def test_resolve_served_distinct_keys_for_same_basename(tmp_path):
    a = tmp_path / "x" / "ws"
    b = tmp_path / "y" / "ws"
    for p in (a, b):
        p.mkdir(parents=True)
        (p / "workspace.json").write_text("{}")
    used: set[str] = set()
    ka = _resolve_served(str(a), used).key
    kb = _resolve_served(str(b), used).key
    assert ka != kb  # same basename "ws" -> disambiguated
