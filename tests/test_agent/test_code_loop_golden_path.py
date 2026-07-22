"""Golden-path smoke for the shipped agent code-loop example.

Locks that ``examples/agent/code_loop_golden_path.py`` — the importable
recipe an agent reimplements via molmcp — still runs end to end (pure Python
API, no LLM loop) and yields one succeeded sweep record per grid point. The
per-behavior locks it leans on (sweep/run semantics, on-disk layout naming,
``add_*`` idempotency) are owned by ``tests/test_workspace`` /
``tests/test_workflow``; this file keeps only the single example-integration
smoke.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from examples.agent.code_loop_golden_path import (
    PARAM_AXIS,
    PARAM_VALUES,
    TASK_NAME,
    run_code_loop_golden_path,
)


class TestCodeLoopGoldenPath:
    def test_yields_one_succeeded_record_per_grid_point(self, tmp_path: Path) -> None:
        records = run_code_loop_golden_path(tmp_path / "ws", plot=False)

        assert len(records) == len(PARAM_VALUES)
        assert all(rec["status"] == "succeeded" for rec in records)
        by_x = {rec[PARAM_AXIS]: rec for rec in records}
        for x in PARAM_VALUES:
            assert by_x[x][TASK_NAME] == pytest.approx(x * x)
            assert by_x[x]["run_id"]
