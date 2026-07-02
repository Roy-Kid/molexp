"""Operator config (``~/.molexp/config.json``) — loader + ``molexp.config`` bridge.

The CLI (``molexp config set agent.model <id>``) persists operator settings to
``~/.molexp/config.json``. Neither application shell may import the other —
so the file loader lives here in the shared :mod:`molexp.services` layer and
both the CLI (:mod:`molexp.cli.config_cmd`) and the server startup bridge
delegate to it, keeping one source of truth for the path, the parsing, and
the ``agent.model`` key name.

:func:`bridge_operator_config` copies the operator-configured values into the
process-global in-code ``molexp.config`` **unless** a value was already
registered in code — in-code configuration always wins. It bridges the agent
model (key ``"agent.model"``) and any ``agent.<provider>_api_key`` entries
(e.g. ``agent.deepseek_api_key`` → ``molexp.config["deepseek_api_key"]``, the
key the router reads). Called at server startup (``create_app``), at plan
preflight (``services.plan_runtime.preflight_plan_router``) and by the agent
CLI — so ``molexp config set agent.deepseek_api_key sk-…`` is a real key path
for every shell, while keys still never come from ``os.environ``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mollog import get_logger

logger = get_logger(__name__)

#: Canonical on-disk operator config file (shared with ``molexp config``).
OPERATOR_CONFIG_PATH = Path.home() / ".molexp" / "config.json"

#: Canonical in-code ``molexp.config`` key for the agent model — same dotted
#: spelling as the CLI key (``molexp config set agent.model <id>``).
AGENT_MODEL_KEY = "agent.model"

#: Legacy flat in-code key, still honoured for backward compatibility.
LEGACY_AGENT_MODEL_KEY = "agent_model"


def load_operator_config(path: Path | None = None) -> dict[str, Any]:
    """Load the operator config file as a plain dict (``{}`` when absent/bad).

    Mirrors the CLI's tolerant behaviour: a missing or unparsable file is an
    empty config, never an exception.
    """
    target = path if path is not None else OPERATOR_CONFIG_PATH
    if not target.exists():
        return {}
    try:
        data = json.loads(target.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"operator config at {target} could not be read; ignoring: {exc}")
        return {}
    return data if isinstance(data, dict) else {}


def configured_agent_model(config: dict[str, Any]) -> str | None:
    """Extract ``agent.model`` from a loaded operator-config dict."""
    agent_section = config.get("agent")
    if isinstance(agent_section, dict):
        model = agent_section.get("model")
        if isinstance(model, str) and model:
            return model
    return None


def configured_api_keys(config: dict[str, Any]) -> dict[str, str]:
    """Extract ``agent.<provider>_api_key`` entries from a loaded config dict.

    Returns ``{"deepseek_api_key": "sk-…", …}`` — the flat spellings the
    router reads from ``molexp.config``. Non-string / empty values are
    ignored.
    """
    agent_section = config.get("agent")
    if not isinstance(agent_section, dict):
        return {}
    return {
        name: value
        for name, value in agent_section.items()
        if name.endswith("_api_key") and isinstance(value, str) and value
    }


def bridge_operator_config(path: Path | None = None) -> None:
    """Bridge operator-config values into the in-code ``molexp.config``.

    Idempotent; safe to call from every shell entry point. Bridges the agent
    model (``agent.model`` → ``molexp.config["agent.model"]``) and every
    ``agent.<provider>_api_key`` entry (→ ``molexp.config["<provider>_api_key"]``).
    A value already registered in code is never overwritten — in-code
    configuration always wins.
    """
    import molexp

    config = load_operator_config(path)

    already = molexp.config.get(AGENT_MODEL_KEY) or molexp.config.get(LEGACY_AGENT_MODEL_KEY)
    if not (isinstance(already, str) and already):
        model = configured_agent_model(config)
        if model is not None:
            molexp.config[AGENT_MODEL_KEY] = model

    for name, value in configured_api_keys(config).items():
        if not molexp.config.get(name):
            molexp.config[name] = value


__all__ = [
    "AGENT_MODEL_KEY",
    "LEGACY_AGENT_MODEL_KEY",
    "OPERATOR_CONFIG_PATH",
    "bridge_operator_config",
    "configured_agent_model",
    "configured_api_keys",
    "load_operator_config",
]
