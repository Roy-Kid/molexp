"""Root-level test configuration.

Hermetic color environment: the CLI tests assert on plain-text output, but
rich obeys ``FORCE_COLOR`` / ``CLICOLOR_FORCE`` from the invoking shell (a
pre-push hook inheriting a developer's ``FORCE_COLOR=3`` embeds ANSI codes
into every CLI capture and breaks those assertions). Normalized HERE, at
conftest import — before any test module can instantiate a rich ``Console``
— so ``pytest tests/`` behaves identically in a bare terminal, a colored
shell, a git hook, and CI. No wrapper env vars needed at any call site.
"""

import os

os.environ.pop("FORCE_COLOR", None)
os.environ.pop("CLICOLOR_FORCE", None)
os.environ["NO_COLOR"] = "1"
