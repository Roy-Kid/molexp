"""Evidence diagnosis: capability-gap classification.

Mirrors ``molexp.harness.stages.evidence`` (``diagnose_failure`` /
``collect_evidence_text``). The one distinction under test: a missing molcrafts
API (``api_symbol_missing`` / confirmed ``SYMBOL_NOT_FOUND``) is a hard
capability gap, while an *invented* ``unknown_capability`` id from a live
catalog — and an environmental ``molmcp_unavailable`` miss — stay retryable.
"""

from __future__ import annotations

import json

from molexp.harness.stages.evidence import collect_evidence_text, diagnose_failure


class TestDiagnoseFailure:
    def test_unknown_capability_is_retryable_not_hard_gap(self) -> None:
        """Binder inventing an id is re-bindable — do not hard-stop the plan."""
        report = {
            "passed": False,
            "violations": [
                {
                    "code": "unknown_capability",
                    "message": "BoundTask 't' references capability_id 'molpy.nope.X' which is not in the registry",
                    "path": "tasks",
                }
            ],
        }
        d = diagnose_failure(Exception("x"), json.dumps(report))
        assert not d.is_capability_gap
        assert "unknown_capability" in d.codes
        assert any("molpy.nope.X" in s for s in d.symbols)


class TestCollectEvidenceText:
    def test_confirmed_symbol_miss_promotes_to_capability_gap(self) -> None:
        d = diagnose_failure(
            Exception("x"),
            "AttributeError: 'molpy.core.box.Box' object has no attribute 'lengths'",
        )
        assert "molpy.core.box.Box.lengths" in d.symbols or any("Box" in s for s in d.symbols)

        def lookup(symbol: str) -> dict:
            return {"ok": False, "code": "SYMBOL_NOT_FOUND", "error": "gone", "ref": symbol}

        text = collect_evidence_text(d, lookup=lookup)
        assert "SYMBOL_NOT_FOUND" in text
        assert d.is_capability_gap

    def test_molmcp_unavailable_is_not_a_capability_gap(self) -> None:
        d = diagnose_failure(Exception("x"), "use molpy.core.box.Box.lengths please")
        text = collect_evidence_text(
            d,
            lookup=lambda s: {
                "ok": False,
                "code": "molmcp_unavailable",
                "error": "no collection",
                "ref": s,
            },
        )
        assert "molmcp_unavailable" in text
        # Environmental miss must not burn the plan as a capability gap.
        assert not d.is_capability_gap
