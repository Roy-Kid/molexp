"""Root-level test configuration.

Hermetic color environment: the CLI tests assert on plain-text output, but
rich/typer help can still embed ANSI (and even split option spellings like
``--workspace`` across SGR codes) when the invoking shell exports
``FORCE_COLOR`` / ``CLICOLOR_FORCE`` / a fancy ``TERM``. Normalized HERE, at
conftest import — before any test module can instantiate a rich ``Console``
— so ``pytest tests/`` behaves identically in a bare terminal, a colored
shell, a git hook, and CI. CI also sets the same vars in the workflow env
block (see ``.github/workflows/ci.yml``).
"""

from __future__ import annotations

import os
import re

os.environ.pop("FORCE_COLOR", None)
os.environ.pop("CLICOLOR_FORCE", None)
os.environ.pop("COLORTERM", None)
os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"
os.environ["FORCE_COLOR"] = "0"

# Shared by CLI assertion helpers — strip SGR even if a Console was created
# before this module ran (plugin import order).
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Return *text* with CSI/SGR color codes removed."""
    return _ANSI_RE.sub("", text)
