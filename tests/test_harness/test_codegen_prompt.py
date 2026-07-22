"""Structured-YAML codegen prompt assembly — ``molexp.harness.prompts.codegen_prompt``."""

from __future__ import annotations

from types import SimpleNamespace

import yaml

from molexp.harness.prompts.codegen_prompt import (
    assemble_task_codegen_prompt,
    assemble_task_test_prompt,
    load_codegen_contract,
    render_contract_yaml,
    wiring_dict_from_bound,
)


class TestCodegenPrompt:
    def test_load_contract_exposes_version_dataflow_and_domain(self) -> None:
        contract = load_codegen_contract()
        assert contract["version"] == "1"
        assert contract["workflow"]["dataflow"]["mode"] == "by_parameter_name"
        assert "molrs" in contract["domain"]["forbid_imports"]
        assert contract["domain"]["frame"] == "molpy.Frame"
        rendered = render_contract_yaml()
        assert (
            "molexp.codegen_contract" in rendered or contract["kind"] == "molexp.codegen_contract"
        )
        assert "RegisterMetric" in rendered

    def test_assemble_codegen_prompt_is_parseable_yaml_document(self) -> None:
        task = SimpleNamespace(
            id="t1",
            ir_task_id="build_monomer",
            capability_id="molpy.foo",
            description="build a monomer",
            inputs={},
            outputs={"monomer": "Frame"},
            parameters={"n": 10},
        )
        edge = SimpleNamespace(source_task_id="t1", target_task_id="t2")
        t2 = SimpleNamespace(
            id="t2",
            ir_task_id="pack",
            capability_id=None,
            description="pack",
            inputs={"monomer": "Frame"},
            outputs={"system": "Frame"},
            parameters={},
        )
        bound = SimpleNamespace(tasks=[task, t2], edges=[edge])
        slugs = {"t1": "build_monomer", "t2": "pack"}
        text = assemble_task_codegen_prompt(
            task=task,
            slug="build_monomer",
            bound=bound,
            slugs=slugs,
            domain_context="# Open molpy.Frame\n...",
            feedback=None,
        )
        doc = yaml.safe_load(text)
        assert doc["role"] == "workflow_task_module"
        assert doc["contract"]["version"] == "1"
        assert doc["task"]["slug"] == "build_monomer"
        assert doc["wiring"]["edges"][0]["shared_keys"] == ["monomer"]
        assert "domain_context" in doc

    def test_assemble_test_prompt_embeds_module_under_test_source(self) -> None:
        task = SimpleNamespace(
            id="t1",
            ir_task_id="build_monomer",
            capability_id=None,
            description=None,
            inputs={},
            outputs={"monomer": "Frame"},
            parameters={},
        )
        text = assemble_task_test_prompt(
            task=task,
            slug="build_monomer",
            module_source="async def build_monomer():\n    return {'monomer': None}\n",
        )
        doc = yaml.safe_load(text)
        assert doc["role"] == "workflow_task_pytest"
        assert "async def build_monomer" in doc["module_under_test"]["source"]

    def test_wiring_dict_shared_keys_is_output_input_intersection(self) -> None:
        t1 = SimpleNamespace(id="a", outputs={"x": 1, "y": 2}, inputs={})
        t2 = SimpleNamespace(id="b", outputs={}, inputs={"x": 1})
        edge = SimpleNamespace(source_task_id="a", target_task_id="b")
        wiring = wiring_dict_from_bound(
            SimpleNamespace(tasks=[t1, t2], edges=[edge]), slugs={"a": "A", "b": "B"}
        )
        assert wiring["edges"][0]["shared_keys"] == ["x"]
