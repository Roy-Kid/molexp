"""Native skill collection.

Each submodule registers its :class:`Skill` instances with
:class:`~molexp.agent.skills.store.SkillStore` at import time. Mirrors
the layout of :mod:`molexp.agent.tools.native`.
"""

from molexp.agent.skills.native import plan

__all__ = ["plan"]
