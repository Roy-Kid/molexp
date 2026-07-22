"""Tests for molexp.profile.models (normalize_profile_name, ProfileConfig, MolCfg)."""

from __future__ import annotations

import pytest

from molexp.profile import MolCfg, ProfileConfig, normalize_profile_name


class TestNormalizeProfileName:
    def test_strips_replaces_dash_preserves_case(self):
        # CLI (--profile dry-run) and YAML keys (dry_run:) unify; casing preserved
        assert normalize_profile_name("  Dry-Run  ") == "Dry_Run"


class TestProfileConfig:
    def test_mapping_access(self):
        cfg = ProfileConfig({"epochs": 10, "dataset": "md17"}, name="dry_run")
        assert cfg["epochs"] == 10
        assert "epochs" in cfg
        assert len(cfg) == 2
        assert set(cfg) == {"epochs", "dataset"}

    def test_construction_deep_copies_input(self):
        data = {"x": 1}
        cfg = ProfileConfig(data, name=None)
        data["x"] = 999
        assert cfg["x"] == 1

    def test_to_dict_returns_deep_copy(self):
        cfg = ProfileConfig({"x": {"y": 1}}, name=None)
        d = cfg.to_dict()
        d["x"]["y"] = 999
        assert cfg["x"]["y"] == 1

    def test_content_hash_ignores_name_and_key_order(self):
        a = ProfileConfig({"x": 1, "y": 2}, name="a")
        b = ProfileConfig({"y": 2, "x": 1}, name="b")
        assert a.content_hash() == b.content_hash()

    def test_content_hash_changes_with_data(self):
        a = ProfileConfig({"x": 1}, name=None)
        b = ProfileConfig({"x": 2}, name=None)
        assert a.content_hash() != b.content_hash()


class TestMolCfgResolve:
    def test_none_returns_defaults_only(self):
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

    def test_dash_key_normalized_at_load_and_resolve(self):
        # field_validator normalizes YAML keys; resolve accepts the CLI dash form
        m = MolCfg.model_validate({"profiles": {"dry-run": {"epochs": 1}}})
        assert "dry_run" in m.profiles
        assert m.resolve("dry-run")["epochs"] == 1

    def test_extends_chain_roots_at_defaults(self):
        m = MolCfg(
            defaults={"epochs": 100, "batch_size": 32, "dataset": "md17"},
            profiles={
                "smoke": {"extends": "defaults", "epochs": 5},
                "tiny_smoke": {"extends": "smoke", "batch_size": 4},
            },
        )
        cfg = m.resolve("tiny_smoke")
        assert cfg["epochs"] == 5  # from smoke
        assert cfg["batch_size"] == 4  # overridden
        assert cfg["dataset"] == "md17"  # from defaults

    def test_deep_merges_nested_keys(self):
        m = MolCfg(
            defaults={"optim": {"lr": 0.001, "momentum": 0.9}},
            profiles={"fast": {"optim": {"lr": 0.01}}},
        )
        cfg = m.resolve("fast")
        assert cfg["optim"] == {"lr": 0.01, "momentum": 0.9}

    def test_unknown_profile_raises_keyerror(self):
        m = MolCfg(profiles={"smoke": {}})
        with pytest.raises(KeyError, match="Unknown profile"):
            m.resolve("missing")

    def test_circular_extends_raises_valueerror(self):
        m = MolCfg(profiles={"a": {"extends": "b"}, "b": {"extends": "a"}})
        with pytest.raises(ValueError, match="Circular"):
            m.resolve("a")
