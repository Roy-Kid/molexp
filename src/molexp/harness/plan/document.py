"""Canonical experiment-plan document (markdown) projection.

The operator-facing **experiment plan book** follows a fixed 12-section outline
(Goal → … → Deliverables → References). This module:

* exposes the outline as :data:`EXPERIMENT_PLAN_OUTLINE` (prompt + docs SoT);
* projects an :class:`ExperimentPlan` (opaque ``spec`` + task board) into filled
  markdown via :func:`render_experiment_plan_document` for review UI / previews
  when no LLM ``body_md`` is available yet.
"""

from __future__ import annotations

from typing import Any

from molexp.harness.plan.experiment_plan import ExperimentPlan
from molexp.harness.plan.task_board import BoardTask, coerce_acceptance

__all__ = [
    "EXPERIMENT_PLAN_OUTLINE",
    "experiment_report_to_document",
    "render_experiment_plan_document",
]

# Outline taught to report writers / plan renderers. Keep section numbers stable.
EXPERIMENT_PLAN_OUTLINE = """\
# Experiment Plan

---

## 1. Goal

**Objective**

> <!-- one-sentence scientific objective -->

---

## 2. Scientific Questions

<!-- what the experiment aims to answer -->

- Q1.
- Q2.
- Q3.

---

## 3. Background

### Current State of the Art

<!-- prior art / current practice -->

### Knowledge Gap

<!-- open problem this experiment addresses -->

### Motivation

<!-- why run this experiment now -->

---

## 4. Hypotheses

### H1

**Description**

<!-- hypothesis -->

**Expected Evidence**

<!-- results that would support it -->

---

### H2

**Description**

<!-- hypothesis -->

**Expected Evidence**

<!-- results that would support it -->

---

## 5. Experimental Design

### System

<!-- system under study -->

### Independent Variables

| Variable | Values | Description |
|----------|--------|-------------|
| | | |

---

### Dependent Variables

| Property | Unit | Description |
|----------|------|-------------|
| | | |

---

### Controlled Variables

| Parameter | Value |
|-----------|-------|
| | |

---

### Computational Methods

| Purpose | Method | Software |
|----------|--------|----------|
| | | |

---

## 6. Workflow

<!-- high-level pipeline; use a text flowchart when helpful -->

```text
Build
    ↓
Optimize
    ↓
Equilibrate
    ↓
Production
    ↓
Analysis
```

---

## 7. Tasks

### Task 1

**Name**

<!-- -->

**Purpose**

<!-- -->

**Inputs**

-

**Outputs**

-

**Method / Tool**

<!-- -->

**Procedure**

1.
2.
3.

**Dependencies**

-

**Acceptance Criteria**

-

---

## 8. Analysis

| Analysis | Input | Output | Purpose |
|----------|-------|--------|---------|
| | | | |

---

## 9. Success Criteria

- [ ]
- [ ]
- [ ]

---

## 10. Risks and Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| | | |

---

## 11. Expected Outcomes

### Expected Results

-

### Alternative Outcomes

-

### Scientific Implications

-

---

## 12. Deliverables

### Data

-

### Figures

-

### Tables

-

### Scripts / Workflow

-

### Final Report

-

---

## References

### Key References

1.
2.
3.

### Related Work

-
"""


def _as_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def _spec_get(spec: dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        raw = spec.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return default


def _table_row(*cells: str) -> str:
    safe = [c.replace("|", "\\|").replace("\n", " ") for c in cells]
    return "| " + " | ".join(safe) + " |"


def _task_section(index: int, task: BoardTask) -> str:
    status = str(task.status.value if hasattr(task.status, "value") else task.status)
    criteria = coerce_acceptance(task.acceptance)
    criteria_lines = "\n".join(f"- {c}" for c in criteria) if criteria else "-"
    deps = getattr(task, "depends_on", None) or ()
    if not deps and hasattr(task, "dependencies"):
        deps = task.dependencies or ()
    dep_list = list(deps) if isinstance(deps, (list, tuple)) else []
    dep_lines = "\n".join(f"- `{d}`" for d in dep_list) if dep_list else "-"
    purpose = ""
    if hasattr(task, "purpose") and task.purpose:
        purpose = str(task.purpose)
    elif hasattr(task, "description") and task.description:
        purpose = str(task.description)
    method = ""
    if hasattr(task, "capability") and task.capability:
        method = str(task.capability)
    elif hasattr(task, "tool") and task.tool:
        method = str(task.tool)
    return "\n".join(
        [
            f"### Task {index}",
            "",
            "**Name**",
            "",
            task.name or task.id,
            "",
            "**Purpose**",
            "",
            purpose or f"_(status: {status})_",
            "",
            "**Inputs**",
            "",
            "-",
            "",
            "**Outputs**",
            "",
            "-",
            "",
            "**Method / Tool**",
            "",
            method or "—",
            "",
            "**Procedure**",
            "",
            "1.",
            "2.",
            "3.",
            "",
            "**Dependencies**",
            "",
            dep_lines,
            "",
            "**Acceptance Criteria**",
            "",
            criteria_lines,
            "",
            "---",
            "",
        ]
    )


def _workflow_ascii(tasks: tuple[BoardTask, ...] | list[BoardTask]) -> str:
    if not tasks:
        return "```text\n(no tasks yet)\n```"
    names = [t.name or t.id for t in tasks]
    body = "\n    ↓\n".join(names)
    return f"```text\n{body}\n```"


def render_experiment_plan_document(
    plan: ExperimentPlan | dict[str, Any],
    *,
    title_fallback: str = "Experiment Plan",
) -> str:
    """Project an experiment plan (spec + board) into the canonical markdown book.

    Fills known sections from the opaque ``spec`` keys and board tasks; leaves
    scientific narrative placeholders only when the plan has no corresponding
    data yet (so the operator still sees the full outline).
    """
    if isinstance(plan, ExperimentPlan):
        spec = dict(plan.spec or {})
        tasks = tuple(plan.board.tasks)
    else:
        spec = dict(plan.get("spec") or {}) if isinstance(plan.get("spec"), dict) else {}
        board = plan.get("board") or {}
        raw_tasks = board.get("tasks") if isinstance(board, dict) else []
        tasks_list: list[BoardTask] = []
        if isinstance(raw_tasks, list):
            for raw in raw_tasks:
                if isinstance(raw, BoardTask):
                    tasks_list.append(raw)
                elif isinstance(raw, dict):
                    try:
                        tasks_list.append(BoardTask.model_validate(raw))
                    except Exception:
                        continue
        tasks = tuple(tasks_list)

    title = _spec_get(spec, "title", "name", "id", default=title_fallback) or title_fallback
    objective = _spec_get(spec, "objective", "goal", "aims")
    system = _spec_get(spec, "system", "system_description", "system_under_study")
    background = _spec_get(spec, "background")
    gap = _spec_get(spec, "knowledge_gap", "gap")
    motivation = _spec_get(spec, "motivation")
    design = _spec_get(spec, "experimental_design", "design", "procedure")
    questions = _as_str_list(spec.get("scientific_questions") or spec.get("questions"))
    hypotheses = _as_str_list(spec.get("hypotheses") or spec.get("scientific_hypothesis"))
    variables = _as_str_list(spec.get("variables") or spec.get("independent_variables"))
    dependents = _as_str_list(spec.get("dependent_variables") or spec.get("observables"))
    controlled = _as_str_list(spec.get("controlled_variables") or spec.get("controlled_conditions"))
    methods = _as_str_list(spec.get("computational_methods") or spec.get("methods"))
    risks = _as_str_list(spec.get("risks") or spec.get("risks_or_uncertainties"))
    expected = _as_str_list(spec.get("expected_outputs") or spec.get("expected_results"))
    success = _as_str_list(spec.get("success_criteria") or spec.get("acceptance"))

    q_lines = (
        "\n".join(
            f"- {q if q.upper().startswith('Q') else f'Q{i}. {q}'}"
            for i, q in enumerate(questions, 1)
        )
        if questions
        else "- Q1. _(to be refined)_\n- Q2.\n- Q3."
    )

    if hypotheses:
        h_blocks: list[str] = []
        for i, h in enumerate(hypotheses, 1):
            h_blocks.append(
                "\n".join(
                    [
                        f"### H{i}",
                        "",
                        "**Description**",
                        "",
                        h,
                        "",
                        "**Expected Evidence**",
                        "",
                        "<!-- results that would support this hypothesis -->",
                        "",
                        "---",
                        "",
                    ]
                )
            )
        hyp_section = "\n".join(h_blocks)
    else:
        hyp_section = "\n".join(
            [
                "### H1",
                "",
                "**Description**",
                "",
                "_(to be refined)_",
                "",
                "**Expected Evidence**",
                "",
                "—",
                "",
                "---",
                "",
            ]
        )

    def _var_table(headers: tuple[str, ...], rows: list[str], ncols: int) -> str:
        head = "| " + " | ".join(headers) + " |"
        sep = "|" + "|".join(["----------"] * ncols) + "|"
        if not rows:
            empty = _table_row(*([""] * ncols))
            return "\n".join([head, sep, empty])
        body = "\n".join(_table_row(r, "", "") if ncols == 3 else _table_row(r, "") for r in rows)
        # Prefer putting full text in first column when we only have names.
        if ncols == 3:
            body = "\n".join(_table_row(r, "—", "—") for r in rows)
        elif ncols == 2:
            body = "\n".join(_table_row(r, "—") for r in rows)
        return "\n".join([head, sep, body])

    indep_table = _var_table(("Variable", "Values", "Description"), variables, 3)
    dep_table = _var_table(("Property", "Unit", "Description"), dependents, 3)
    ctrl_table = _var_table(("Parameter", "Value"), controlled, 2)
    method_table = _var_table(("Purpose", "Method", "Software"), methods, 3)

    if tasks:
        task_blocks = "\n".join(_task_section(i, t) for i, t in enumerate(tasks, 1))
    else:
        task_blocks = "_(no tasks on the board yet)_\n\n---\n"

    risk_rows = (
        "\n".join(_table_row(r, "—", "—") for r in risks) if risks else _table_row("", "", "")
    )
    success_lines = "\n".join(f"- [ ] {s}" for s in success) if success else "- [ ]\n- [ ]\n- [ ]"
    expected_lines = "\n".join(f"- {e}" for e in expected) if expected else "-"

    parts = [
        f"# Experiment Plan: {title}",
        "",
        "---",
        "",
        "## 1. Goal",
        "",
        "**Objective**",
        "",
        f"> {objective}" if objective else "> _(objective not set)_",
        "",
        "---",
        "",
        "## 2. Scientific Questions",
        "",
        q_lines,
        "",
        "---",
        "",
        "## 3. Background",
        "",
        "### Current State of the Art",
        "",
        background or "_(to be refined)_",
        "",
        "### Knowledge Gap",
        "",
        gap or "_(to be refined)_",
        "",
        "### Motivation",
        "",
        motivation or objective or "_(to be refined)_",
        "",
        "---",
        "",
        "## 4. Hypotheses",
        "",
        hyp_section.rstrip(),
        "",
        "## 5. Experimental Design",
        "",
        "### System",
        "",
        system or design or "_(to be refined)_",
        "",
        "### Independent Variables",
        "",
        indep_table,
        "",
        "---",
        "",
        "### Dependent Variables",
        "",
        dep_table,
        "",
        "---",
        "",
        "### Controlled Variables",
        "",
        ctrl_table,
        "",
        "---",
        "",
        "### Computational Methods",
        "",
        method_table,
        "",
        "---",
        "",
        "## 6. Workflow",
        "",
        design or "High-level pipeline derived from the task board:",
        "",
        _workflow_ascii(tasks),
        "",
        "---",
        "",
        "## 7. Tasks",
        "",
        task_blocks.rstrip(),
        "",
        "## 8. Analysis",
        "",
        "| Analysis | Input | Output | Purpose |",
        "|----------|-------|--------|---------|",
        "| | | | |",
        "",
        "---",
        "",
        "## 9. Success Criteria",
        "",
        success_lines,
        "",
        "---",
        "",
        "## 10. Risks and Mitigation",
        "",
        "| Risk | Impact | Mitigation |",
        "|------|--------|------------|",
        risk_rows,
        "",
        "---",
        "",
        "## 11. Expected Outcomes",
        "",
        "### Expected Results",
        "",
        expected_lines,
        "",
        "### Alternative Outcomes",
        "",
        "-",
        "",
        "### Scientific Implications",
        "",
        "-",
        "",
        "---",
        "",
        "## 12. Deliverables",
        "",
        "### Data",
        "",
        "-",
        "",
        "### Figures",
        "",
        "-",
        "",
        "### Tables",
        "",
        "-",
        "",
        "### Scripts / Workflow",
        "",
        "-",
        "",
        "### Final Report",
        "",
        "-",
        "",
        "---",
        "",
        "## References",
        "",
        "### Key References",
        "",
        "1.",
        "2.",
        "3.",
        "",
        "### Related Work",
        "",
        "-",
        "",
    ]
    return "\n".join(parts).rstrip() + "\n"


def experiment_report_to_document(report: object) -> str:
    """Prefer ``body_md`` on an ExperimentReport-like object; else reconstruct.

    Args:
        report: :class:`ExperimentReport` instance or a mapping with the same keys.
    """
    if report is None:
        return ""
    if isinstance(report, dict):
        body = report.get("body_md")
        if isinstance(body, str) and body.strip():
            return body.strip() + ("\n" if not body.endswith("\n") else "")
        title = str(report.get("title") or "Experiment Plan")
        objective = str(report.get("objective") or "")
        system = str(report.get("system_description") or "")
        design = str(report.get("experimental_design") or "")
        background = report.get("background")
        hyp = report.get("scientific_hypothesis")
        variables = _as_str_list(report.get("variables"))
        controlled = _as_str_list(report.get("controlled_conditions"))
        expected = _as_str_list(report.get("expected_outputs"))
        risks = _as_str_list(report.get("risks_or_uncertainties"))
    else:
        body = getattr(report, "body_md", None)
        if isinstance(body, str) and body.strip():
            return body.strip() + ("\n" if not body.endswith("\n") else "")
        title = str(getattr(report, "title", None) or "Experiment Plan")
        objective = str(getattr(report, "objective", None) or "")
        system = str(getattr(report, "system_description", None) or "")
        design = str(getattr(report, "experimental_design", None) or "")
        background = getattr(report, "background", None)
        hyp = getattr(report, "scientific_hypothesis", None)
        variables = _as_str_list(getattr(report, "variables", None))
        controlled = _as_str_list(getattr(report, "controlled_conditions", None))
        expected = _as_str_list(getattr(report, "expected_outputs", None))
        risks = _as_str_list(getattr(report, "risks_or_uncertainties", None))

    # Reconstruct a thin document from legacy structured fields.
    pseudo = {
        "spec": {
            "title": title,
            "objective": objective,
            "system_description": system,
            "experimental_design": design,
            "background": background,
            "scientific_hypothesis": hyp,
            "variables": variables,
            "controlled_conditions": controlled,
            "expected_outputs": expected,
            "risks_or_uncertainties": risks,
        },
        "board": {"tasks": []},
    }
    return render_experiment_plan_document(pseudo, title_fallback=title)
