# Server Lifecycle

`molexp` has two ways to run the FastAPI server:

1. **`molexp serve` CLI** — foreground uvicorn; the simplest path and the one you'll use for dev and most deployments.
2. **`ServerManager` Python API** (`molexp.server.manager`) — programmatic lifecycle (start / stop / status / logs, background daemon, kill-on-exit) for embedding the server inside a test harness or tooling.

Pick the foreground CLI unless you specifically need a daemonized process.

## `molexp serve` (foreground CLI)

```bash
molexp serve --workspace ./lab --port 8000 --host localhost
```

- Resolves the workspace directory (auto-detects a `./workspace/` subfolder if present, warns if no `workspace.json` is found).
- `cd`s into the workspace so relative paths inside user scripts work.
- Detects the bundled SPA at `src/molexp/_webapp/` via `importlib.resources`; if empty, runs **API-only** and prints instructions for the Vite dev server.
- Runs `uvicorn.run(app, host=..., port=..., log_level="info")` inline (foreground, blocks until `Ctrl+C`).

### Serving several workspaces

`molexp serve` accepts repeated `--workspace` / `-ws` options. The first
workspace is active at startup; the full served set is exposed at
`GET /api/workspaces`, and the UI can switch the active workspace through
`POST /api/workspace/open`.

```bash
molexp serve \
  --workspace /Users/roykid/work/molcrafts/molexp \
  --workspace /Users/roykid/work/molcrafts/polymer_electrolyte \
  --port 8000
```

The server assigns each entry a stable key such as `local-molexp` or
`local-polymer_electrolyte`. The aggregate surface under
`/api/workspaces/{key}/...` lets the UI list projects from several served
workspaces without ID collisions. The existing flat routes, such as
`/api/projects` and `/api/workspace/runs`, continue to address the active
workspace so single-workspace clients keep working unchanged.

Remote workspace specs (`user@host:/path` or `@target-name`) are listed in the
same served set. Unreachable remotes remain visible in `GET /api/workspaces`
with `unreachable: true`; read routes return a remote-unreachable error instead
of failing the whole workspace list, and mutating scoped remote routes are
rejected as read-only.

### Watching run progress

For a long-running workspace such as
`/Users/roykid/work/molcrafts/polymer_electrolyte`, serve that workspace and
run the workflow in another terminal:

```bash
molexp serve --workspace /Users/roykid/work/molcrafts/polymer_electrolyte --port 8000
```

```bash
cd /Users/roykid/work/molcrafts/polymer_electrolyte
python build_flow.py
```

Open the bundled UI, or run the Vite dev server if the backend reports
API-only mode. The Runs view polls `/api/workspace/runs` every three seconds
through a shared frontend store, so new runs, execution attempts, status
changes, scheduler metadata, and completion/failure state appear without a page
reload. The header shows the last sync time and the refresh button triggers an
immediate fetch. When several workspaces are served, make
`polymer_electrolyte` the first `--workspace` or activate it in the left
workspace tree before watching its run dashboard.

This is a thin wrapper. There is no `--dev` / `--reload` flag in `molexp serve` today — for hot reload, invoke uvicorn directly:

```bash
uvicorn molexp.server.app:app --reload --port 8000
```

## `ServerManager` (Python API)

`molexp.server.manager.ServerManager` is a lifecycle helper kept around for programmatic use (e.g. integration tests that want a real server running in the background).

```python
from molexp.server import ServerManager

manager = ServerManager()

# Start (foreground)
manager.start(port=8000, dev=True)

# Start (background daemon; persists after the Python process exits)
manager.start(background=True, kill_on_exit=False)

# Start (background, auto-killed when the Python process exits — useful in tests)
manager.start(background=True, kill_on_exit=True)

# Check status
manager.status()        # → {"api": {...}, "ui": {...}}
manager.is_running()

# Stream logs
for line in manager.get_logs(lines=50, follow=False):
    print(line)

# Stop
manager.stop(ui=True)
```

### `start()` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `host` | `"0.0.0.0"` | Host to bind |
| `port` | `8000` | Port |
| `dev` | `True` | Pass `--reload` to uvicorn |
| `background` | `False` | Run as a subprocess (daemon) |
| `ui` | `False` | Also start `npm run dev` in `ui/` |
| `sample_data` | `False` | Run `create_sample_data.py` first (legacy helper) |
| `kill_on_exit` | `False` | When `background=True`, tie subprocess lifetime to the parent process (keeps it in the same PG and registers `atexit` + signal handlers) |

### PID and log files

`ServerManager` stores PIDs and logs under `~/.molexp/`:

```
~/.molexp/
├── server.pid      ← API server PID
├── ui.pid          ← UI dev server PID (if started)
└── logs/
    ├── server.log
    └── ui.log
```

Pass a custom `config_dir=Path("./.local")` to the constructor to relocate them.

### Use Cases

| Scenario | Pattern |
|----------|---------|
| Local dev | `molexp serve` in one terminal, `npm run dev` in another |
| Production (long-lived) | `manager.start(background=True, kill_on_exit=False)` from a deploy script |
| Tests (auto-cleanup) | `manager.start(background=True, kill_on_exit=True)` |
| Embedded tooling | Run `molexp.server.app:create_app()` directly in your own ASGI host |

## Bundled UI Detection

`create_app()` looks for the SPA bundle via:

```python
from importlib.resources import files
webapp = files("molexp") / "_webapp"
if webapp.is_dir() and (webapp / "index.html").exists():
    mount(app, webapp)
```

This works for editable installs, wheels, and packaged releases. The bundle is populated by `npm run build:ui` before `python -m build --wheel`. If it is empty (typical dev), the server runs API-only with a `/` fallback advertising `/api/docs` and `/api/health`.

## Troubleshooting

- **Port busy.** `ServerManager.start()` raises `RuntimeError: Server is already running` if `~/.molexp/server.pid` points at a live process. Call `manager.stop()` first or `rm` the stale pid file.
- **API-only despite a build.** Check that `src/molexp/_webapp/index.html` exists in your active installation. Editable installs re-use the in-tree bundle; wheels ship a frozen copy.
- **Background process won't die.** Confirm `kill_on_exit=True` — with `False`, the subprocess is intentionally detached (`start_new_session=True`) and must be stopped via `manager.stop()`.

## Runnable Example

`examples/operations/server_lifecycle.py` spawns the API server through `ServerManager`, polls `status()`, and stops it cleanly — the minimal programmatic lifecycle.
