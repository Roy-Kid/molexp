"""Skill subsystem — saved behaviour bundles, slash-invokable.

Public surface:

- :class:`Skill` — the saved-behaviour record.
- :class:`SkillStore` — three-tier (native + user + workspace) store.
- :class:`ParsedCommand` / :func:`parse` — slash-command parser keyed
  off the store's slash names.

Importing this subpackage triggers :mod:`molexp.agent.skills.native`
so the in-process registrations tier (``Scope.NATIVE``) is populated.
"""

# Side-effect: register every native skill on SkillStore at import time.
from molexp.agent.skills import native as _native  # noqa: F401
from molexp.agent.skills.commands import (
    CommandKind,
    ParsedCommand,
    parse,
)
from molexp.agent.skills.store import (
    SKILLS_FILE,
    USER_HOME_DIR_NAME,
    USER_HOME_SKILLS_FILE,
    SkillStore,
)
from molexp.agent.skills.types import (
    RESERVED_SLASH_NAMES,
    SLASH_NAME_RE,
    Skill,
)

__all__ = [
    "CommandKind",
    "ParsedCommand",
    "RESERVED_SLASH_NAMES",
    "SKILLS_FILE",
    "SLASH_NAME_RE",
    "Skill",
    "SkillStore",
    "USER_HOME_DIR_NAME",
    "USER_HOME_SKILLS_FILE",
    "parse",
]
