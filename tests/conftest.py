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
from collections.abc import Iterator

import pytest

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


@pytest.fixture(autouse=True)
def _hermetic_operator_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[None]:
    """Isolate every test from the developer's ``~/.molexp/config.json``.

    Two leaks, one fixture:

    * **read** — ``bridge_operator_config`` (called by every preflight) loads
      the operator config from ``OPERATOR_CONFIG_PATH``; pointed at an empty
      tmp file so a laptop with a real key/model does not silently satisfy
      tests that assert a *missing* key or an unknown model.
    * **write** — the bridge's destination, ``molexp.config``, is a
      **process-global** singleton that no monkeypatch unwinds. One test that
      bridges a configured ``agent.models`` map used to re-tier every later
      test in the same process (an unknown-model preflight quietly resolved to
      the laptop's DeepSeek models and stopped raising). Snapshot + restore.
    """
    import molexp
    from molexp.services import operator_config

    monkeypatch.setattr(
        operator_config,
        "OPERATOR_CONFIG_PATH",
        tmp_path_factory.mktemp("operator-config") / "config.json",
    )
    before = dict(molexp.config)
    yield
    for key in list(molexp.config.keys()):
        if key not in before:
            del molexp.config[key]
    for key, value in before.items():
        molexp.config[key] = value
