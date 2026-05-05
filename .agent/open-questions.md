# Open Questions

Bootstrap-time uncertainties recorded by `/mol-agent:bootstrap`. Add a
date and resolve each one as the answer becomes clear.

## 2026-05-05 — molexp-specific axes the generic mol plugin does not cover

Three project-local agents were retired together with the `/molexp-*`
skill suite:

- `molexp-designer` — UI visual quality, information density, design
  tokens, accessibility (no equivalent in the `mol` plugin's
  agent set).
- `molexp-integrity` — experiment reproducibility, atomic-write
  correctness, param-space determinism, concurrent-run safety
  (closest generic agent: `mol:scientist`, but it targets numerical
  correctness, not workflow integrity).
- `molexp-security` — prompt-injection surfaces, secret handling,
  API auth, LLM output sanitization, FastAPI input validation
  (no equivalent in the `mol` plugin).

Decide whether to:
1. propose these as generic `mol` plugin agents (designer, integrity,
   security) so other mol-family projects benefit, or
2. reintroduce them locally if and when the gap actually bites.

The retired agent definitions can be recovered from git history if
needed (commit `d05fe29` and earlier).
