"""Public-API regression for plan-bundle-01-run."""

from __future__ import annotations

import molexp.harness as harness
from molexp.harness import Plan


def main() -> None:
    assert "Plan" in harness.__all__
    assert "run_plan" not in harness.__all__
    assert "PlanOrchestrator" not in harness.__all__
    assert isinstance(Plan, type)
    try:
        import molexp.harness.modes.plan_orchestrator  # noqa: F401
    except ModuleNotFoundError:
        print("plan-bundle-01-run: ok")
        return
    raise AssertionError("plan_orchestrator module must be gone")


if __name__ == "__main__":
    main()
