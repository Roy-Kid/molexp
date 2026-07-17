"""Plugin authoring — ``CliPlugin`` construction and ``SubmitHandler`` composition.

Matches ``docs/concept/plugins.md``.

Demonstrates:

1. ``SubmitHandler`` composition — the pattern used by ``--scheduler slurm``.
2. The ``CliPlugin`` API surface — register a handler, expose it to the CLI.
3. How a plugin bridges scheduler-specific args into the workflow execution path.

This example shows the *construction pattern* — a real plugin would live in its
own package with a proper ``plugin.json`` + entry point. The scheduler_molq.py
example shows the same pattern for Slurm; here we show a simplified custom one.

Run directly::

    python examples/plugins/custom_submit_handler.py
"""

from __future__ import annotations


def main() -> None:
    # ── 1. SubmitHandler is a typed Protocol (from workspace layer) ─────
    # In a real plugin, you'd import and implement:
    #
    #   from molexp.workspace.handler import SubmitHandler
    #
    #   class CustomSubmitHandler(SubmitHandler):
    #       def submit(self, *, execution: Execution, target: ComputeTarget) -> JobRef:
    #           ...

    print("SubmitHandler Protocol shape:")
    print("  submit(execution, target) -> JobRef")
    print("  cancel(job_ref) -> None")
    print()

    # ── 2. CliPlugin construction pattern ────────────────────────────────
    # A real plugin registers through the ``molexp.cli_plugins`` entry point:
    #
    #   [project.entry-points."molexp.cli_plugins"]
    #   custom = "molexp_custom_scheduler.plugin:CliPlugin"
    #
    # The plugin's CliPlugin instance exposes:
    #   - name: str          — plugin identifier
    #   - submit_handler     — the SubmitHandler implementation
    #   - cli_group          — optional typer command group for CLI args

    print("CliPlugin registration pattern:")
    print("  1. Define a SubmitHandler subclass")
    print("  2. Create a CliPlugin instance wrapping it")
    print("  3. Register via pyproject.toml entry point: molexp.cli_plugins")
    print("  4. CLI discovers it: molexp run --scheduler custom")
    print()

    # ── 3. Example: minimal CliPlugin ────────────────────────────────────
    print("Example minimal plugin skeleton:")
    print()
    print("  # my_scheduler/plugin.py")
    print("  from molexp.plugins.cli import CliPlugin")
    print("  from molexp.workspace.handler import SubmitHandler")
    print()
    print("  class MySubmitHandler(SubmitHandler):")
    print("      def submit(self, execution, target):")
    print("          print(f'Submitting {execution.id} to {target.host}')")
    print("          return JobRef(id='job-123')")
    print()
    print("  plugin = CliPlugin(name='custom', handler=MySubmitHandler())")
    print()

    print("Done — plugin authoring pattern demonstrated.")


if __name__ == "__main__":
    main()
