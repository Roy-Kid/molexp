"""Tests for molexp.config.models (ProfileConfig, MolCfg)."""

from __future__ import annotations

import pytest

from molexp.config import MolCfg, ProfileConfig, normalize_profile_name


class TestNormalization:
    def test_dash_to_underscore(self):
        assert normalize_profile_name("dry-run") == "dry_run"

    def test_strips_whitespace(self):
        assert normalize_profile_name("  smoke  ") == "smoke"

    def test_preserves_case(self):
        assert normalize_profile_name("Fast-Mode") == "Fast_Mode"


class TestProfileConfig:
    def test_mapping_access(self):
        cfg = ProfileConfig({"epochs": 10, "dataset": "md17"}, name="dry_run")
        assert cfg["epochs"] == 10
        assert cfg.get("dataset") == "md17"
        assert cfg.get("missing", "fallback") == "fallback"
        assert "epochs" in cfg
        assert len(cfg) == 2
        assert set(cfg) == {"epochs", "dataset"}

    def test_name_normalized_on_construction(self):
        cfg = ProfileConfig({}, name="dry-run")
        assert cfg.name == "dry_run"

    def test_none_name(self):
        cfg = ProfileConfig({"x": 1}, name=None)
        assert cfg.name is None

    def test_immutable(self):
        data = {"x": 1}
        cfg = ProfileConfig(data, name=None)
        data["x"] = 999
        assert cfg["x"] == 1  # deep-copied on construction

    def test_content_hash_deterministic(self):
        a = ProfileConfig({"x": 1, "y": 2}, name="a")
        b = ProfileConfig({"y": 2, "x": 1}, name="b")  # different name & order
        # hash ignores profile name and key order
        assert a.content_hash() == b.content_hash()

    def test_content_hash_changes_with_data(self):
        a = ProfileConfig({"x": 1}, name=None)
        b = ProfileConfig({"x": 2}, name=None)
        assert a.content_hash() != b.content_hash()

    def test_to_dict_returns_copy(self):
        cfg = ProfileConfig({"x": {"y": 1}}, name=None)
        d = cfg.to_dict()
        d["x"]["y"] = 999
        assert cfg["x"]["y"] == 1


class TestMolCfgResolve:
    def test_resolve_none_returns_defaults(self):
        m = MolCfg(defaults={"epochs": 100, "dataset": "md17"})
        cfg = m.resolve(None)
        assert cfg.name is None
        assert cfg["epochs"] == 100

    def test_profile_overrides_defaults(self):
        m = MolCfg(
            defaults={"epochs": 100, "dataset": "md17"},
            profiles={"dry_run": {"epochs": 1}},
        )
        cfg = m.resolve("dry_run")
        assert cfg.name == "dry_run"
        assert cfg["epochs"] == 1
        assert cfg["dataset"] == "md17"  # inherited from defaults

    def test_profile_name_dash_accepted(self):
        m = MolCfg(profiles={"dry_run": {"x": 1}})
        cfg = m.resolve("dry-run")  # CLI form
        assert cfg.name == "dry_run"

    def test_yaml_dash_key_normalized_at_load(self):
        # Profile keys with dashes are normalized when the model is built
        m = MolCfg.model_validate(
            {"profiles": {"dry-run": {"epochs": 1}}}
        )
        assert "dry_run" in m.profiles
        assert m.resolve("dry-run")["epochs"] == 1

    def test_extends_chain(self):
        m = MolCfg(
            defaults={"epochs": 100, "batch_size": 32, "dataset": "md17"},
            profiles={
                "smoke": {"extends": "defaults", "epochs": 5},
                "tiny_smoke": {"extends": "smoke", "batch_size": 4},
            },
        )
        cfg = m.resolve("tiny_smoke")
        assert cfg["epochs"] == 5       # from smoke
        assert cfg["batch_size"] == 4   # overridden
        assert cfg["dataset"] == "md17"  # from defaults

    def test_unknown_profile_raises(self):
        m = MolCfg(profiles={"smoke": {}})
        with pytest.raises(KeyError, match="Unknown profile"):
            m.resolve("missing")

    def test_circular_extends_raises(self):
        m = MolCfg(
            profiles={
                "a": {"extends": "b"},
                "b": {"extends": "a"},
            }
        )
        with pytest.raises(ValueError, match="Circular"):
            m.resolve("a")

    def test_deep_merge(self):
        m = MolCfg(
            defaults={"optim": {"lr": 0.001, "momentum": 0.9}},
            profiles={"fast": {"optim": {"lr": 0.01}}},
        )
        cfg = m.resolve("fast")
        # deep-merge preserves untouched nested keys
        assert cfg["optim"] == {"lr": 0.01, "momentum": 0.9}
