"""Frozen-API contract checks of ``TestSourceValidator`` (``_contract_violations``).

These are the static gates that catch generated-test mistakes which would
always fail at pytest time: a bare ``ANY`` with no import, and ``.name`` access
on ``RegisterMetric`` (whose field is ``.key``). Each check is exercised with
its rejecting case and its allowed boundary.
"""

from __future__ import annotations

from molexp.harness.validators.test_source import TestSourceValidator


class TestTestSourceValidatorContract:
    def test_bare_any_without_import_is_rejected(self) -> None:
        src = """\
from unittest.mock import patch

def test_pack():
    with patch("workflow.pack.pack", return_value=ANY):
        pass
"""
        report = TestSourceValidator.validate(src)
        assert report.passed is False
        assert any(v.code == "undefined_any" for v in report.violations)

    def test_any_with_import_passes(self) -> None:
        src = """\
from unittest.mock import ANY, patch

def test_pack_box_ok():
    with patch("workflow.pack.pack", return_value=ANY):
        assert True
"""
        report = TestSourceValidator.validate(src)
        assert report.passed is True

    def test_hasattr_name_on_register_metric_is_rejected(self) -> None:
        src = """\
from molexp.workflow import RegisterMetric

def test_energy():
    m = RegisterMetric(key="e", value=1.0)
    assert hasattr(m, "name")
"""
        report = TestSourceValidator.validate(src)
        assert report.passed is False
        assert "register_metric_name_attr" in {v.code for v in report.violations}

    def test_register_metric_key_access_passes(self) -> None:
        src = """\
from molexp.workflow import RegisterMetric

def test_metric_key():
    m = RegisterMetric(key="final_energy", value=-1.0)
    assert m.key == "final_energy"
    assert m.value == -1.0
"""
        report = TestSourceValidator.validate(src)
        assert report.passed is True
