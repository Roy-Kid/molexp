"""Static contract checks on generated test source (RegisterMetric / ANY)."""

from __future__ import annotations

from molexp.harness.validators.test_source import TestSourceValidator


def test_rejects_register_metric_name_attr() -> None:
    src = """\
from molexp.workflow import RegisterMetric

def test_energy():
    m = RegisterMetric(key="e", value=1.0)
    assert hasattr(m, "name")
"""
    report = TestSourceValidator.validate(src)
    assert report.passed is False
    codes = {v.code for v in report.violations}
    assert "register_metric_name_attr" in codes


def test_rejects_bare_any_without_import() -> None:
    src = """\
from unittest.mock import patch

def test_pack():
    with patch("workflow.pack.pack", return_value=ANY):
        pass
"""
    report = TestSourceValidator.validate(src)
    assert report.passed is False
    assert any(v.code == "undefined_any" for v in report.violations)


def test_allows_any_when_imported() -> None:
    src = """\
from unittest.mock import ANY, patch

def test_pack_box_ok():
    with patch("workflow.pack.pack", return_value=ANY):
        assert True
"""
    report = TestSourceValidator.validate(src)
    assert report.passed is True


def test_allows_register_metric_key() -> None:
    src = """\
from molexp.workflow import RegisterMetric

def test_metric_key():
    m = RegisterMetric(key="final_energy", value=-1.0)
    assert m.key == "final_energy"
    assert m.value == -1.0
"""
    report = TestSourceValidator.validate(src)
    assert report.passed is True
