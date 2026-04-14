"""Tests for molexp.config.loader."""

from __future__ import annotations

import json

import pytest

from molexp.config import load_molcfg
from molexp.config.loader import find_default_config


YAML_SAMPLE = """\
version: 1
defaults:
  dataset: md17
  epochs: 100

profiles:
  dry-run:
    extends: defaults
    epochs: 1
  smoke:
    extends: defaults
    epochs: 5
"""


class TestLoadMolCfg:
    def test_load_yaml(self, tmp_path):
        p = tmp_path / "molcfg.yaml"
        p.write_text(YAML_SAMPLE)
        cfg = load_molcfg(p)
        assert cfg.defaults == {"dataset": "md17", "epochs": 100}
        # dashes normalized at load
        assert "dry_run" in cfg.profiles
        resolved = cfg.resolve("dry-run")
        assert resolved.name == "dry_run"
        assert resolved["epochs"] == 1
        assert resolved["dataset"] == "md17"

    def test_load_json(self, tmp_path):
        p = tmp_path / "molcfg.json"
        p.write_text(
            json.dumps(
                {
                    "defaults": {"x": 1},
                    "profiles": {"quick": {"x": 2}},
                }
            )
        )
        cfg = load_molcfg(p)
        assert cfg.resolve("quick")["x"] == 2

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_molcfg(tmp_path / "no.yaml")

    def test_unsupported_suffix(self, tmp_path):
        p = tmp_path / "cfg.toml"
        p.write_text("x = 1\n")
        with pytest.raises(ValueError, match="Unsupported"):
            load_molcfg(p)


class TestFindDefault:
    def test_finds_yaml(self, tmp_path):
        (tmp_path / "molcfg.yaml").write_text("defaults: {}\n")
        assert find_default_config(tmp_path) == tmp_path / "molcfg.yaml"

    def test_returns_none_when_absent(self, tmp_path):
        assert find_default_config(tmp_path) is None

    def test_prefers_yaml_over_json(self, tmp_path):
        (tmp_path / "molcfg.yaml").write_text("defaults: {}\n")
        (tmp_path / "molcfg.json").write_text("{}")
        assert find_default_config(tmp_path).name == "molcfg.yaml"
