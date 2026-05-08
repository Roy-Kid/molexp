"""``AgentRunner`` + ``PlanMode`` — plan a parameter sweep, run it, aggregate.

Plan mode is more than a one-shot prompt: it drives a multi-stage
``molexp.workflow`` (intake → goal → context → method → decomposition →
protocol → preview → gate-A → task-binding → compose → compile →
dry-run → gate-B → handoff) and surfaces the structured plan via
``AgentRunResult.mode_state["plan"]``. To make every stage *visible*, this
example frames a real research task — screening reaction conditions for a
Suzuki-Miyaura coupling over a solvent x base x temperature grid — and
shows how the plan-mode output is consumed downstream:

1. The user describes the system + parameter grid in natural language.
2. ``PlanMode`` produces a structured plan (``intake``, ``goal``,
   ``decomposition``, ``protocol``, …) and a final handoff summary.
3. A toy yield model simulates each condition (mock execution).
4. The results are aggregated into an experiment report.

Uses pydantic-ai's built-in ``"test"`` model so the example runs offline.
The test model returns canned text, so the value here is the *shape* of
plan mode — swap the model string for ``"openai:gpt-4o-mini"`` (with
``OPENAI_API_KEY`` set) to see real planning content.

Run directly::

    python examples/agent/plan_mode.py
"""

from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass
from itertools import product
from typing import Any

from molexp.agent import AgentRunner, AgentSession
from molexp.agent.modes import PlanMode

SYSTEM_BRIEF = (
    "Suzuki-Miyaura coupling between 4-bromoanisole and phenylboronic acid. "
    "Goal: maximize isolated yield by screening solvent, base, and temperature."
)

PARAMETER_GRID: dict[str, list[Any]] = {
    "solvent": ["toluene", "dioxane", "ethanol_water_4_1"],
    "base": ["K2CO3", "Cs2CO3", "K3PO4"],
    "temp_C": [70, 85, 100],
}


@dataclass(frozen=True)
class Condition:
    solvent: str
    base: str
    temp_C: int

    def label(self) -> str:
        return f"{self.solvent:<18} | {self.base:<7} | {self.temp_C:>3} C"


@dataclass(frozen=True)
class Aggregate:
    n_conditions: int
    best: tuple[Condition, float]
    worst: tuple[Condition, float]
    mean_yield: float
    by_solvent_mean: dict[str, float]
    by_base_mean: dict[str, float]
    by_temp_mean: dict[int, float]


def enumerate_conditions() -> list[Condition]:
    return [
        Condition(s, b, t)
        for s, b, t in product(
            PARAMETER_GRID["solvent"],
            PARAMETER_GRID["base"],
            PARAMETER_GRID["temp_C"],
        )
    ]


def simulate_yield(c: Condition, *, rng: random.Random) -> float:
    """Toy model — peaks near dioxane / K3PO4 / 85 C with mild noise."""
    s_pref = {"toluene": 0.80, "dioxane": 1.00, "ethanol_water_4_1": 0.70}[c.solvent]
    b_pref = {"K2CO3": 0.85, "Cs2CO3": 0.95, "K3PO4": 1.00}[c.base]
    t_pref = math.exp(-((c.temp_C - 85) ** 2) / (2 * 8 ** 2))
    raw = 95.0 * s_pref * b_pref * t_pref + rng.gauss(0.0, 1.5)
    return max(0.0, min(99.0, raw))


def aggregate(rows: list[tuple[Condition, float]]) -> Aggregate:
    rows_sorted = sorted(rows, key=lambda r: r[1], reverse=True)
    yields = [y for _, y in rows]

    def group_mean(attr: str) -> dict[Any, float]:
        groups: dict[Any, list[float]] = {}
        for c, y in rows:
            groups.setdefault(getattr(c, attr), []).append(y)
        return {k: sum(v) / len(v) for k, v in groups.items()}

    return Aggregate(
        n_conditions=len(rows),
        best=rows_sorted[0],
        worst=rows_sorted[-1],
        mean_yield=sum(yields) / len(yields),
        by_solvent_mean=group_mean("solvent"),
        by_base_mean=group_mean("base"),
        by_temp_mean=group_mean("temp_C"),
    )


def render_plan(plan: dict[str, Any], summary: str) -> str:
    """Render the workflow stages exposed in ``mode_state['plan']``."""
    sections = {
        "intake": plan.get("intake"),
        "goal": plan.get("goal"),
        "context": plan.get("context"),
        "decomposition": plan.get("decomposition"),
        "protocol": plan.get("protocol"),
        "design": plan.get("design"),
        "compile_status": plan.get("compile_status"),
        "dry_run_status": plan.get("dry_run_status"),
        "iterations": plan.get("iterations"),
        "handoff": summary,
    }
    lines: list[str] = []
    for stage, body in sections.items():
        lines.append(f"\n[{stage}]")
        lines.append(str(body).strip() if body else "(empty)")
    return "\n".join(lines)


def render_report(rows: list[tuple[Condition, float]], agg: Aggregate) -> str:
    out: list[str] = []
    out.append("solvent             | base    | temp  | yield")
    out.append("-" * 52)
    for c, y in rows:
        out.append(f"{c.label()} | {y:5.1f}%")
    out.append("-" * 52)
    out.append(f"n conditions : {agg.n_conditions}")
    out.append(f"best         : {agg.best[0].label()} -> {agg.best[1]:.1f}%")
    out.append(f"worst        : {agg.worst[0].label()} -> {agg.worst[1]:.1f}%")
    out.append(f"mean yield   : {agg.mean_yield:.1f}%")
    out.append("by solvent (mean):")
    for s, m in agg.by_solvent_mean.items():
        out.append(f"  - {s:<18} {m:5.1f}%")
    out.append("by base (mean):")
    for b, m in agg.by_base_mean.items():
        out.append(f"  - {b:<18} {m:5.1f}%")
    out.append("by temperature (mean):")
    for t, m in agg.by_temp_mean.items():
        out.append(f"  - {t:>3} C{'':<14} {m:5.1f}%")
    return "\n".join(out)


def banner(title: str) -> str:
    return "\n" + "=" * 60 + f"\n{title}\n" + "=" * 60


async def main() -> None:
    print(banner("EXPERIMENT BRIEF"))
    print(SYSTEM_BRIEF)
    print("\nParameter grid:")
    for k, v in PARAMETER_GRID.items():
        print(f"  {k:<8}: {v}")

    runner = AgentRunner(mode=PlanMode(max_iterations=4), model="test")
    session = AgentSession()
    user_request = (
        f"{SYSTEM_BRIEF}\n\n"
        "Plan a full-factorial screening campaign over the grid below, then "
        "describe how to aggregate the per-condition yields:\n"
        f"  solvent in {PARAMETER_GRID['solvent']}\n"
        f"  base    in {PARAMETER_GRID['base']}\n"
        f"  temp_C  in {PARAMETER_GRID['temp_C']}\n"
        "Required outputs: per-axis mean yields and the global optimum."
    )
    result = await runner.run(session, user_request)

    print(banner("PLAN MODE OUTPUT"))
    print(f"summary text  : {result.text[:120]}")
    print(f"history turns : {len(result.messages)}")
    plan = (result.mode_state or {}).get("plan", {})
    print(render_plan(plan, result.text))

    print(banner("EXECUTING SWEEP (mock yield model)"))
    rng = random.Random(0)
    rows = [(c, simulate_yield(c, rng=rng)) for c in enumerate_conditions()]
    print(f"ran {len(rows)} conditions")

    print(banner("AGGREGATED REPORT"))
    print(render_report(rows, aggregate(rows)))


if __name__ == "__main__":
    asyncio.run(main())
