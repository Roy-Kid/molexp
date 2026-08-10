# Server Lifecycle

`molexp` has two ways to run the FastAPI server:

1. **`molexp serve` CLI** — foreground uvicorn; the simplest path and the one you'll use for dev and most deployments.
2. **`ServerManager` Python API** (`molexp.server.manager`) — programmatic lifecycle (start / stop / status / logs, background daemon, kill-on-exit) for embedding the server inside a test harness or tooling.

Pick the foreground CLI unless you specifically need a daemonized process.

## `molexp serve` (foreground CLI)

```bash
molexp serve -ws ./lab --port 8000 --host localhost
```

### HTTP auth (optional)

By default the API is open on loopback. To require login for the UI and
`/api/*` (filesystem users under `~/.molexp/auth/`):

```bash
# Bootstrap the default admin (password via prompt or MOLEXP_AUTH_PASSWORD)
molexp auth login -u admin

molexp serve -ws ./lab --auth
# optional: require a specific account to exist
molexp serve -ws ./lab --auth -u admin
```

CLI surface mirrors `gh auth`: `login` / `logout` / `status` / `switch` /
`token` / `refresh`, plus `molexp auth users …` for the local user table.
Binding a non-loopback `--host` without `--auth` (or `auth.enabled` in
`~/.molexp/config.json`) is refused at startup.

Admin users can manage accounts from **Settings → Users** in the UI (shadcn
table + dialogs; data via TanStack Query).

| Config / flag | Meaning |
|---|---|
| `--auth` / `auth.enabled: true` | Require login for `/api/*` (except status/login/health) |
| `-u` / `--user` | Require that username already exists at serve start |
| `auth.session_ttl_hours` | Session lifetime (default **168** = 7 days) |

Failed logins are rate-limited in-process (5 failures / 5 minutes → 5 minute
lockout per username and client host).

- Resolves the local workspace with `pathlib.Path`; if the directory or
  `workspace.json` is missing, initializes the workspace at that exact path.
- Activates the workspace through the server's path override without changing
  the process working directory.
- Detects the bundled SPA at `src/molexp/_webapp/` via `importlib.resources`; if empty, runs **API-only** and prints instructions for the Vite dev server.
- Runs `uvicorn.run(app, host=..., port=..., log_level="info")` inline (foreground, blocks until `Ctrl+C`).

### Serving several workspaces

`molexp serve` accepts repeated `-ws` / `--workspace` options. The first
workspace is active at startup; the full served set is exposed at
`GET /api/workspaces`, and the UI can switch the active workspace through
`POST /api/workspace/open`.

```bash
molexp serve \
  -ws /Users/roykid/work/molcrafts/molexp \
  -ws /Users/roykid/work/molcrafts/polymer_electrolyte \
  --port 8000
```

The server assigns each entry a stable key such as `local-molexp` or
`local-polymer_electrolyte`. The aggregate surface under
`/api/workspaces/{key}/...` lets the UI list projects from several served
workspaces without ID collisions. The existing flat routes, such as
`/api/projects` and `/api/workspace/runs`, continue to address the active
workspace so single-workspace clients keep working unchanged.

`serve` no longer accepts compute-target aliases or SSH workspace specs. Remote
compute targets remain execution concerns and are configured independently;
the server's `-ws` inputs are local workspace paths only.

### Watching run progress

For a long-running workspace such as
`/Users/roykid/work/molcrafts/polymer_electrolyte`, serve that workspace and
run the workflow in another terminal:

```bash
molexp serve -ws /Users/roykid/work/molcrafts/polymer_electrolyte --port 8000
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
`polymer_electrolyte` the first `-ws` value or activate it in the left
workspace tree before watching its run dashboard.

### Frontend HMR (`--dev`)

From a source checkout (editable install), start the API and the Rsbuild
dev server together:

```bash
molexp serve --dev -ws ./lab --port 8000
# optional: --ui-port 5173
```

- Spawns the `ui` leaf script
  `MOLEXP_API_PORT=<api-port> npm run dev:api -- --port=<ui-port>`
  (repo root: `npm run dev:api`) so `/api` on the UI origin proxies to
  this process. Offline mock UI is `npm run dev:ui` (MSW), not this path.
- Prints **Dev UI** `http://localhost:<ui-port>` — open that for live
  reload. The API port still serves the bundled `dist` if present; that
  is the production package, not HMR.
- Ctrl+C stops both processes (UI process group is SIGTERM'd).
- Needs `npm` on PATH and `ui/package.json`. Set `MOLEXP_UI_DIR` if the
  ui tree is not next to the package sources.

For Python-only auto-reload of the API (no UI), invoke uvicorn directly:

```bash
uvicorn --factory molexp.server.app:create_app --reload --port 8000
```

## `ServerManager` (Python API)

`molexp.server.manager.ServerManager` is a lifecycle helper kept around for programmatic use (e.g. integration tests that want a real server running in the background).

```python
# docs: skip — starts live server processes (ports, pid files, daemons)
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
| `ui` | `False` | Also start the `ui` leaf `npm run dev:api` (root: `npm run dev:api`) |
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
| Local dev | `molexp serve --dev -ws ./lab` (API + UI HMR); or two terminals without `--dev` |
| Production (long-lived) | `manager.start(background=True, kill_on_exit=False)` from a deploy script |
| Tests (auto-cleanup) | `manager.start(background=True, kill_on_exit=True)` |
| Embedded tooling | Run `molexp.server.app:create_app()` directly in your own ASGI host |

## Bundled UI Detection

`create_app()` looks for the SPA bundle via:

```python
# docs: skip — illustrative fragment (``mount``/``app`` are create_app internals)
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
