"""Operator config (``~/.molexp/config.json``) — loader + ``molexp.config`` bridge.

The CLI (``molexp config set agent.model <id>``) persists operator settings to
``~/.molexp/config.json``. The server, by layer rule, must not import
``molexp.cli`` — so the file loader lives here and the CLI delegates to it
(:mod:`molexp.cli.config_cmd`), keeping one source of truth for the path,
the parsing, and the ``agent.model`` key name.

At server startup :func:`bridge_operator_config` copies the operator-configured
agent model into the process-global in-code ``molexp.config`` (key
``"agent.model"``) **unless** a value was already registered in code — in-code
configuration always wins.
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


def bridge_operator_config(path: Path | None = None) -> None:
    """Bridge operator-config values into the in-code ``molexp.config``.

    Called once at server startup (``create_app``). Currently bridges only the
    agent model: ``agent.model`` from ``~/.molexp/config.json`` is copied into
    ``molexp.config["agent.model"]`` when neither the canonical nor the legacy
    in-code key is already set, so code-registered values keep precedence.
    """
    import molexp

    already = molexp.config.get(AGENT_MODEL_KEY) or molexp.config.get(LEGACY_AGENT_MODEL_KEY)
    if isinstance(already, str) and already:
        return
    model = configured_agent_model(load_operator_config(path))
    if model is not None:
        molexp.config[AGENT_MODEL_KEY] = model


__all__ = [
    "AGENT_MODEL_KEY",
    "LEGACY_AGENT_MODEL_KEY",
    "OPERATOR_CONFIG_PATH",
    "bridge_operator_config",
    "configured_agent_model",
    "load_operator_config",
]
