# Writing a third-party molexp plugin

molexp ships **two completely independent** plugin channels so a
downstream Python package can extend molexp without forking it or
re-bundling the frontend:

| Channel | What it adds | Runs in | Entry-point group |
|---|---|---|---|
| **CLI** | New subcommands under ``molexp <yourcmd>`` | Python process at startup | ``molexp.cli_plugins`` |
| **UI** | Dynamic-imported React bundle in the SPA | Browser, on demand | ``molexp.ui_plugins`` |

A package may contribute either channel, both, or neither. The two
channels do **not** share an entry point, a contract, or an API
version — they have completely different runtimes, lifecycles, and
evolution cadences, so they evolve independently.

> Why two? CLI lives in the Python process from boot and binds to
> Typer. UI lives in the browser, is fetched as ESM only when needed,
> and binds to the React contribution runtime. The only thing they
> have in common is that pip is the distribution channel — so each
> uses its own ``[project.entry-points]`` group in your
> ``pyproject.toml``.

## Adding a CLI command

A CLI plugin is a single :class:`molexp.plugins.cli.CliPlugin` instance
referenced from the ``molexp.cli_plugins`` entry-point group.

```toml
# pyproject.toml
[project]
name = "molexp-plugin-greeter"
version = "0.1.0"
dependencies = ["molexp", "typer"]

[project.entry-points."molexp.cli_plugins"]
greeter = "my_plugin.cli:plugin"
```

```python
# my_plugin/cli.py
import typer

from molexp.plugins.cli import CliPlugin


def _hello(name: str = "world") -> None:
    typer.echo(f"hello, {name}")


def _register(app: typer.Typer) -> None:
    app.command(name="greet")(_hello)


plugin = CliPlugin(
    id="greeter",
    name="Greeter",
    version="0.1.0",
    register=_register,
)
```

After ``pip install`` of this package alongside molexp:

```bash
$ molexp greet --name Alice
hello, Alice
```

### CLI contract

| Field | Type | Required | Purpose |
|---|---|---|---|
| ``id`` | ``str`` | yes | Globally unique short identifier |
| ``name`` | ``str`` | yes | Human-readable plugin name |
| ``version`` | ``str`` | yes | Plugin's own semver |
| ``register`` | ``Callable[[typer.Typer], None]`` | yes | Attach commands to the molexp Typer app |
| ``api_version`` | ``str`` | no — defaults to ``"1"`` | CLI contract version targeted |

The instance is **frozen**. ``register`` has no ``None`` default — a
package without CLI contributions has no business in this group.

The current expected version is
``molexp.plugins.cli.CLI_PLUGIN_API_VERSION``. Plugins targeting a
mismatched version are silently skipped at discovery time (logged as a
warning), and a single broken ``register`` call will not prevent the
rest of the CLI from booting — molexp catches per-plugin exceptions
and keeps going.

## Adding a UI bundle

A UI plugin is just **a directory** that contains a pre-built ESM
bundle and a ``manifest.json`` describing it. Python's only role is to
tell molexp's server where the directory lives — all UI semantics
(``id``, ``name``, ``version``, ``api_version``, entry point,
capabilities) are declared in ``manifest.json``, which is fetched by
the browser-side loader.

### 1. Declare the entry point

```toml
# pyproject.toml
[project.entry-points."molexp.ui_plugins"]
greeter = "my_plugin.ui:bundle_dir"
```

The referenced symbol must be **either** a :class:`pathlib.Path`
**or** a zero-argument callable returning ``Path``. The
entry-point name (``greeter`` above) becomes the plugin id used in the
mount URL.

```python
# my_plugin/ui.py
from pathlib import Path


def bundle_dir() -> Path:
    """Return the directory containing manifest.json + index.js."""
    return Path(__file__).parent / "ui_dist"
```

### 2. Ship a ``manifest.json``

The bundle directory must contain a ``manifest.json`` at its root. The
schema (validated by the browser loader against
``UiBundleManifest`` in ``ui/src/plugins/types.ts``):

```json
{
  "id": "greeter",
  "name": "Greeter",
  "version": "0.1.0",
  "api_version": "1",
  "entry": "index.js",
  "capabilities": ["greeter"]
}
```

| Field | Type | Required | Purpose |
|---|---|---|---|
| ``id`` | ``string`` | yes | Must match the entry-point name |
| ``name`` | ``string`` | yes | Human-readable plugin name |
| ``version`` | ``string`` | yes | Plugin's own semver |
| ``api_version`` | ``"1"`` | yes | UI plugin contract version |
| ``entry`` | ``string`` | no — defaults to ``"index.js"`` | Filename of the ESM entry, relative to bundle root |
| ``capabilities`` | ``string[]`` | no | Free-form tags for downstream consumers |

The browser checks ``manifest.api_version`` against the
``UI_PLUGIN_API_VERSION`` constant frozen into the SPA build at compile
time. Bundles with a mismatched version are skipped (logged via
``console.warn``), and the rest of the SPA continues loading.

### 3. Ship a default-exporting ``index.js``

```javascript
// my_plugin/ui_dist/index.js
const greeter = {
  id: "greeter",
  register() {
    // Use @/app/registry / @/plugins/contribution-runtime APIs to
    // contribute renderers, file-preview plugins, execution columns,
    // etc. The same APIs built-in plugins use.
  },
};

export default greeter;
```

The bundle must be a valid native ESM module (``<script type="module">``
form). The default export must match the ``UiPluginModule`` shape:
``{ id: string, register: () => void | Promise<void> }``. Bundle React
and any other dependencies — the file is served as a static asset and
must be fully self-contained. Use whatever build tool you like
(rsbuild / vite / rollup) so long as the output is a single ESM file.

### How it gets served

molexp's FastAPI server discovers your bundle via the entry-point
group, mounts the directory at ``/api/plugins/<id>/``, and surfaces the
descriptor through ``GET /api/plugins`` as
``{id, manifestUrl: "/api/plugins/<id>/manifest.json", entryUrl:
"/api/plugins/<id>/index.js"}``. The browser loader then:

1. fetches ``manifestUrl`` and validates the body;
2. checks ``manifest.api_version`` matches the build constant;
3. dynamic-imports ``entryUrl`` (or the override from
   ``manifest.entry``) and calls ``register()`` on the default export.

Per-plugin failures at every step are isolated with ``console.warn``
so a single broken bundle cannot prevent other plugins from loading.

## Why CLI and UI are independent

Both channels are discovered through Python ``importlib.metadata``
entry points, but **that's the only similarity**.

- They live in different runtimes (Python process vs. browser).
- They load at different times (process startup vs. browser idle tick).
- They depend on different libraries (Typer vs. React + your bundler).
- Their API versions evolve on different cadences. ``CLI_PLUGIN_API_VERSION``
  and ``UI_PLUGIN_API_VERSION`` are independent constants — bumping one
  does not force the other.
- Python carries **zero** UI semantics: no ``UiPlugin`` dataclass, no
  ``api_version`` field, no manifest parsing. UI semantics live in
  TS-side types and the bundle's ``manifest.json``. Python is the
  distribution shim only.

This separation is deliberate: a single ``MolexpPlugin`` mega-class
that bundled both would force a Python release just because the UI
manifest schema changed (or vice versa), which is exactly the wrong
coupling.

## Failure isolation

Every callback you provide runs inside a try/except (Python) or
try/catch (TypeScript) frame. A single broken plugin will not prevent
molexp itself from starting, nor will it block other plugins from
loading. Failures are logged as warnings; if your plugin disappears,
check ``stderr`` (Python) or the browser console (UI) for a
``[plugins]`` line.

## Security boundary

**Plugins run in the host molexp process with full privileges.** The
mechanism is a discovery convention, not a sandbox: there is no
permissions model, no per-plugin filesystem jail, and no network
restriction. Treat plugin packages exactly the way you would treat any
other PyPI dependency:

* only install plugins from sources you trust;
* pin versions in your environment so a compromised release of a
  plugin you already trust cannot silently auto-upgrade;
* review the CLI ``register`` callable and the UI ``index.js`` before
  deploying a new plugin to a production workspace.

If you need stronger isolation (e.g. for hosted multi-tenant
deployments), run molexp itself in a confined process and treat the
plugin allowlist as part of your deployment manifest.

## Built-in plugins are not third-party

molexp ships a few in-tree plugins (``core``, ``metrics``, ``molq``,
``molvis``) that are statically imported by ``App.tsx`` and live
inside the main bundle. They do **not** appear in ``GET /api/plugins``
and do **not** participate in the entry-point discovery described
here. Conversely, third-party packages cannot use those reserved ids.
The two paths run side-by-side without conflict.
