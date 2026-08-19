"""Plugin host: Context, Host, service keys, profile composers.

Import ``molexp.harness.host`` — this package is not re-exported from
``molexp.harness`` (frozen 22-symbol public surface).
"""

from molexp.harness.errors import PluginInjectError
from molexp.harness.host.compose import compose_chat, compose_curate, compose_plan, compose_run
from molexp.harness.host.context import Context
from molexp.harness.host.host import Host
from molexp.harness.host.keys import Keys
from molexp.harness.host.plugin import Plugin

__all__ = [
    "Context",
    "Host",
    "Keys",
    "Plugin",
    "PluginInjectError",
    "compose_chat",
    "compose_curate",
    "compose_plan",
    "compose_run",
]
