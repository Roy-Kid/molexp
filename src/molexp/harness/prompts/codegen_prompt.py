"""Structured YAML prompts for per-task workflow/test codegen.

Plan agents receive a **document** (YAML), not a wall of English prose:

* ``contract`` — frozen molexp L0 surface (:file:`contracts/molexp_codegen.v1.yaml`)
* ``task`` / ``wiring`` — this bound task + inter-task IO map
* ``repair`` / ``evidence`` — optional revision payload
* ``domain_context`` — optional short molmcp-compose excerpt

System prompts stay one-line roles; the YAML is the user payload.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "assemble_task_codegen_prompt",
    "assemble_task_test_prompt",
    "load_codegen_contract",
    "render_contract_yaml",
    "task_dict_from_bound",
    "wiring_dict_from_bound",
]

_CONTRACT_PATH = Path(__file__).resolve().parent / "contracts" / "molexp_codegen.v1.yaml"


@lru_cache(maxsize=1)
def load_codegen_contract() -> dict[str, Any]:
    """Load the versioned molexp codegen contract (L0 primitives)."""
    data = yaml.safe_load(_CONTRACT_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"codegen contract must be a mapping: {_CONTRACT_PATH}")
    return data


def render_contract_yaml() -> str:
    """YAML dump of the contract only (for docs / legacy system prompts)."""
    return yaml.safe_dump(
        load_codegen_contract(),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )


def task_dict_from_bound(task: object, *, slug: str) -> dict[str, Any]:
    """Project a bound-task object into a compact YAML-friendly mapping."""
    ins = getattr(task, "inputs", None) or {}
    outs = getattr(task, "outputs", None) or {}
    params = getattr(task, "parameters", None) or {}
    return {
        "id": getattr(task, "id", None),
        "ir_task_id": getattr(task, "ir_task_id", None) or getattr(task, "id", None),
        "slug": slug,
        "capability_id": getattr(task, "capability_id", None),
        "description": getattr(task, "description", None),
        "inputs": _mappingish(ins),
        "outputs": _mappingish(outs),
        "parameters": _mappingish(params),
    }


def wiring_dict_from_bound(bound: object, *, slugs: dict[str, str]) -> dict[str, Any]:
    """Inter-task dataflow map (molexp engine semantics — not from molmcp)."""
    tasks = list(getattr(bound, "tasks", ()) or ())
    edges = list(getattr(bound, "edges", ()) or ())
    by_id = {t.id: t for t in tasks}
    edge_list: list[dict[str, Any]] = []
    for e in edges:
        src = getattr(e, "source_task_id", None)
        tgt = getattr(e, "target_task_id", None)
        if src not in by_id or tgt not in by_id:
            continue
        src_outs = _mappingish(getattr(by_id[src], "outputs", None) or {})
        tgt_ins = _mappingish(getattr(by_id[tgt], "inputs", None) or {})
        # Keys that appear on both sides are the contract surface.
        shared = sorted(set(src_outs) & set(tgt_ins)) if src_outs and tgt_ins else []
        edge_list.append(
            {
                "from_task": slugs.get(src, src),
                "to_task": slugs.get(tgt, tgt),
                "from_outputs": src_outs,
                "to_inputs": tgt_ins,
                "shared_keys": shared or "match_by_name",
            }
        )
    task_rows = []
    for t in tasks:
        tid = t.id
        task_rows.append(
            {
                "slug": slugs.get(tid, tid),
                "id": tid,
                "inputs": _keys(getattr(t, "inputs", None) or {}),
                "outputs": _keys(getattr(t, "outputs", None) or {}),
            }
        )
    return {
        "rule": (
            "Each task returns a dict; keys become the next task's parameter "
            "names. Types must agree across edges. molmcp does not document this."
        ),
        "tasks": task_rows,
        "edges": edge_list,
    }


def assemble_task_codegen_prompt(
    *,
    task: object,
    slug: str,
    bound: object | None = None,
    slugs: dict[str, str] | None = None,
    experiment_spec: dict[str, Any] | str | None = None,
    domain_context: str | None = None,
    feedback: str | None = None,
    catalog_excerpt: str | None = None,
) -> str:
    """Build the user payload YAML for ``workflow_source_file_writer``."""
    doc: dict[str, Any] = {
        "role": "workflow_task_module",
        "instruction": (
            "Emit ONE WorkflowSource with files=[workflow/<slug>.py] only. "
            "async def named exactly <slug>. Follow contract + wiring. "
            "No prose, no markdown fences."
        ),
        "contract": load_codegen_contract(),
        "task": task_dict_from_bound(task, slug=slug),
    }
    if bound is not None and slugs is not None:
        doc["wiring"] = wiring_dict_from_bound(bound, slugs=slugs)
    if experiment_spec is not None:
        doc["experiment_spec"] = experiment_spec
    if catalog_excerpt:
        doc["capability_catalog_excerpt"] = _clip(catalog_excerpt, 4000)
    if domain_context:
        doc["domain_context"] = _clip(domain_context, 6000)
    if feedback:
        doc["repair"] = {"feedback": _clip(feedback, 8000)}
    return _dump(doc)


def assemble_task_test_prompt(
    *,
    task: object,
    slug: str,
    module_source: str,
    bound: object | None = None,
    slugs: dict[str, str] | None = None,
    domain_context: str | None = None,
    feedback: str | None = None,
) -> str:
    """Build the user payload YAML for ``test_code_file_writer``."""
    doc: dict[str, Any] = {
        "role": "workflow_task_pytest",
        "instruction": (
            f"Emit ONE TestSource with files=[tests/test_{slug}.py]. "
            f"Import `from workflow.{slug} import {slug}`. "
            "Test the real public contract; mock only symbols the module imports. "
            "No prose, no markdown fences."
        ),
        "contract": load_codegen_contract(),
        "task": task_dict_from_bound(task, slug=slug),
        "module_under_test": {
            "path": f"workflow/{slug}.py",
            "source": module_source,
        },
    }
    if bound is not None and slugs is not None:
        doc["wiring"] = wiring_dict_from_bound(bound, slugs=slugs)
    if domain_context:
        doc["domain_context"] = _clip(domain_context, 6000)
    if feedback:
        doc["repair"] = {"feedback": _clip(feedback, 8000)}
    return _dump(doc)


def _dump(doc: dict[str, Any]) -> str:
    return yaml.safe_dump(
        doc,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=100,
    )


def _clip(text: str, n: int) -> str:
    text = text.strip()
    if len(text) <= n:
        return text
    return text[: n - 20] + "\n# … truncated …\n"


def _mappingish(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): _jsonish(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonish(x) for x in obj]
    return _jsonish(obj)


def _keys(obj: object) -> list[str]:
    if isinstance(obj, dict):
        return [str(k) for k in obj]
    if isinstance(obj, (list, tuple)):
        return [str(x) for x in obj]
    return []


def _jsonish(v: object) -> object:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, dict):
        return {str(k): _jsonish(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonish(x) for x in v]
    # pydantic models etc.
    dump = getattr(v, "model_dump", None)
    if callable(dump):
        try:
            return _jsonish(dump(mode="json"))
        except Exception:
            pass
    return str(v)
