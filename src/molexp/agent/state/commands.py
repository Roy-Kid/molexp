"""Slash command parser: turns raw chat text into a structured intent.

Two flavours:

1. **Builtin commands** — handled by the chat client itself, never reach
   the LLM. Reserved names: ``/plan``, ``/clear``, ``/model``, ``/help``.

2. **Skill commands** — invocations of saved :class:`~.skills.Skill`
   templates with non-empty ``slash_name``. Arguments are ``key=value``
   pairs (shell-quoted values supported via :mod:`shlex`).

The parser is intentionally a pure function over text + a
:class:`SkillStore` snapshot — no I/O of its own — so the same routine
can run server-side (route handler) and client-side via the parse
endpoint, keeping a single source of truth for grammar and errors.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Literal

from molexp.agent.state.skills import RESERVED_SLASH_NAMES, SkillStore

CommandKind = Literal["skill", "builtin", "error"]


@dataclass(frozen=True)
class ParsedCommand:
    """Outcome of :func:`parse` for a single chat input.

    ``kind`` discriminates the union:

    - ``"skill"`` — ``skill_id``, ``parameters``, ``plan_mode`` populated.
    - ``"builtin"`` — ``name`` is the builtin's slug; ``parameters`` carries
      raw key/value args; ``plan_mode`` is True for ``/plan``.
    - ``"error"`` — ``error`` carries a UI-ready message; ``name`` echoes
      the slash name when known.
    """

    kind: CommandKind
    name: str = ""
    skill_id: str = ""
    parameters: dict[str, str] = field(default_factory=dict)
    plan_mode: bool = False
    error: str = ""


def parse(raw: str, store: SkillStore) -> ParsedCommand:
    """Parse a single line of chat input.

    Returns a :class:`ParsedCommand`. Non-slash inputs and malformed
    input both yield ``kind="error"`` rather than raising — the chat
    handler is expected to surface ``error`` directly to the user.
    """
    text = raw.strip()
    if not text.startswith("/"):
        return ParsedCommand(
            kind="error",
            error="Slash commands must start with '/'.",
        )

    body = text[1:]
    if not body:
        return ParsedCommand(kind="error", error="Empty slash command.")

    try:
        tokens = shlex.split(body, posix=True)
    except ValueError as exc:
        return ParsedCommand(
            kind="error",
            error=f"Could not parse arguments: {exc}",
        )
    if not tokens:
        return ParsedCommand(kind="error", error="Empty slash command.")

    head, *args = tokens
    name = head.lower()

    parameters, parse_error = _parse_kv_args(args)
    if parse_error is not None:
        return ParsedCommand(kind="error", name=name, error=parse_error)

    if name in RESERVED_SLASH_NAMES:
        return _builtin(name, parameters)

    skill = store.find_by_slash(name)
    if skill is None:
        return ParsedCommand(
            kind="error",
            name=name,
            error=f"Unknown command '/{name}'. Define a skill with this slash name first.",
        )

    required = skill.required_parameters()
    missing = [k for k in required if k not in parameters]
    if missing:
        return ParsedCommand(
            kind="error",
            name=name,
            skill_id=skill.id,
            error=(
                f"Missing required parameter(s) for /{name}: "
                + ", ".join(sorted(missing))
            ),
        )

    return ParsedCommand(
        kind="skill",
        name=name,
        skill_id=skill.id,
        parameters=parameters,
        plan_mode=skill.default_plan_mode,
    )


def _parse_kv_args(tokens: list[str]) -> tuple[dict[str, str], str | None]:
    """Convert ``["key=value", "k2=v2"]`` into a dict.

    Returns ``(params, error)``. A token without ``=`` produces an
    error rather than being silently dropped — this is friendlier
    than letting a typo silently succeed.
    """
    params: dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            return params, (
                f"Argument '{token}' is missing a value. Use the form key=value."
            )
        key, _, value = token.partition("=")
        key = key.strip()
        if not key:
            return params, f"Empty argument name in '{token}'."
        params[key] = value
    return params, None


def _builtin(name: str, parameters: dict[str, str]) -> ParsedCommand:
    """Construct a :class:`ParsedCommand` for a builtin slash command.

    The runtime semantics live in the UI; the parser only marks
    ``plan_mode=True`` for ``/plan`` so the chat client can flip its
    next-message toggle without re-parsing.
    """
    return ParsedCommand(
        kind="builtin",
        name=name,
        parameters=parameters,
        plan_mode=(name == "plan"),
    )
