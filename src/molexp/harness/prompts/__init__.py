"""Shipped system prompts for the harness's LLM-driven planning agents.

Each planning stage (`GenerateExperimentReport`, `ExtractWorkflowIR`,
`BindMolcraftsTasks`, `GenerateTestSpec`) delegates to an
:class:`~molexp.harness.gateways.gateway.AgentGateway` keyed by ``agent_name``;
this package holds the real, shipped system prompt for each. They are honest
domain role descriptions — they tell the model what to produce, never how to
dodge a validator. Plain-string modules only (no ``pydantic_ai`` /
``pydantic_graph`` import), so importing this package keeps the harness
import-guard green.

Wire them into a gateway through its ``system_prompt_by_agent`` argument::

    from molexp.harness.prompts import prompts_by_agent

    gateway = RouterBackedAgentGateway(..., system_prompt_by_agent=prompts_by_agent())

Production model wiring (used by the live example in
``plan-mode-revival-04``)::

    from molexp.agent.router import ModelTier
    from molexp.agent._pydanticai.router import PydanticAIRouter

    router = PydanticAIRouter(models=dict.fromkeys(ModelTier, "deepseek:deepseek-v4-flash"))
"""

from __future__ import annotations

from molexp.harness.prompts.bound_workflow import SYSTEM_PROMPT as BOUND_WORKFLOW_SYSTEM_PROMPT
from molexp.harness.prompts.experiment_report import (
    SYSTEM_PROMPT as EXPERIMENT_REPORT_SYSTEM_PROMPT,
)
from molexp.harness.prompts.test_spec import SYSTEM_PROMPT as TEST_SPEC_SYSTEM_PROMPT
from molexp.harness.prompts.workflow_ir import SYSTEM_PROMPT as WORKFLOW_IR_SYSTEM_PROMPT

__all__ = [
    "BOUND_WORKFLOW_SYSTEM_PROMPT",
    "EXPERIMENT_REPORT_SYSTEM_PROMPT",
    "TEST_SPEC_SYSTEM_PROMPT",
    "WORKFLOW_IR_SYSTEM_PROMPT",
    "prompts_by_agent",
]


def prompts_by_agent() -> dict[str, str]:
    """Return the shipped system prompt for each planning ``agent_name``.

    Keys match the ``agent_name``s the harness planning stages set on their
    :class:`~molexp.harness.schemas.AgentCallSpec`; pass the result as a
    gateway's ``system_prompt_by_agent``.
    """
    return {
        "experiment_report_writer": EXPERIMENT_REPORT_SYSTEM_PROMPT,
        "workflow_ir_extractor": WORKFLOW_IR_SYSTEM_PROMPT,
        "bound_workflow_binder": BOUND_WORKFLOW_SYSTEM_PROMPT,
        "test_spec_writer": TEST_SPEC_SYSTEM_PROMPT,
    }
