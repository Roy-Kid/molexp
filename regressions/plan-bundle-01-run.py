"""Public-API regression for plan-bundle-01-run."""

from __future__ import annotations

import molexp.harness as harness


def main() -> None:
    assert "run_plan" in harness.__all__
    assert "PlanOrchestrator" not in harness.__all__
    assert callable(harness.run_plan)
    try:
        import molexp.harness.modes.plan_orchestrator  # noqa: F401
    except ModuleNotFoundError:
        print("plan-bundle-01-run: ok")
        return
    raise AssertionError("plan_orchestrator module must be gone")


if __name__ == "__main__":
    main()
