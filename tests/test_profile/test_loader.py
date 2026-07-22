"""Tests for molexp.profile.loader (load_molcfg, find_default_config)."""

from __future__ import annotations

import json

import pytest

from molexp.profile import load_molcfg
from molexp.profile.loader import find_default_config

YAML_SAMPLE = """\
version: 1
defaults:
  dataset: md17
  epochs: 100

profiles:
  dry-run:
    extends: defaults
    epochs: 1
"""


class TestLoadMolCfg:
    def test_yaml_suffix_parses_into_model(self, tmp_path):
        p = tmp_path / "molcfg.yaml"
        p.write_text(YAML_SAMPLE)
        cfg = load_molcfg(p)
        assert cfg.defaults == {"dataset": "md17", "epochs": 100}
        assert "dry_run" in cfg.profiles  # dash key normalized at load

    def test_json_suffix_parses_into_model(self, tmp_path):
        p = tmp_path / "molcfg.json"
        p.write_text(json.dumps({"defaults": {"x": 1}, "profiles": {"quick": {"x": 2}}}))
        cfg = load_molcfg(p)
        assert cfg.defaults == {"x": 1}
        assert cfg.profiles == {"quick": {"x": 2}}

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_molcfg(tmp_path / "no.yaml")

    def test_unsupported_suffix_raises(self, tmp_path):
        p = tmp_path / "cfg.toml"
        p.write_text("x = 1\n")
        with pytest.raises(ValueError, match="Unsupported"):
            load_molcfg(p)


class TestFindDefaultConfig:
    def test_returns_none_when_absent(self, tmp_path):
        assert find_default_config(tmp_path) is None

    def test_prefers_yaml_over_json(self, tmp_path):
        (tmp_path / "molcfg.yaml").write_text("defaults: {}\n")
        (tmp_path / "molcfg.json").write_text("{}")
        assert find_default_config(tmp_path).name == "molcfg.yaml"
