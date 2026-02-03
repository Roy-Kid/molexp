# Server Lifecycle Management

`molexp` provides flexible control over server process lifecycle through the `ServerManager` class.

## Background Process Behavior

When starting servers in background mode, you can control whether they should continue running after the main process exits:

### Default Behavior (Daemon Mode)

```python
from molexp.server import ServerManager

manager = ServerManager()
pids = manager.start(
    background=True,
    kill_on_exit=False,  # Default: processes continue running
)
# Server continues running even after script exits
```

In this mode:
- Background processes are started in a new session (`start_new_session=True`)
- They become daemon processes independent of the parent
- They persist after the main process exits
- Must be stopped manually with `manager.stop()`

### Kill on Exit Mode

```python
from molexp.server import ServerManager

manager = ServerManager()
pids = manager.start(
    background=True,
    kill_on_exit=True,  # NEW: kill when parent exits
)
# Server will be automatically killed when script exits
```

In this mode:
- Background processes remain in the same process group
- They are automatically killed when the main process exits
- Cleanup handlers are registered for graceful shutdown
- Useful for temporary servers or testing

## API Reference

### ServerManager.start()

```python
def start(
    self,
    host: str = "0.0.0.0",
    port: int = 8000,
    dev: bool = True,
    background: bool = False,
    ui: bool = False,
    sample_data: bool = False,
    kill_on_exit: bool = False,  # NEW parameter
) -> dict[str, int]:
```

#### Parameters

- **host** (`str`): Host address to bind to
- **port** (`int`): Port number for API server
- **dev** (`bool`): Enable development mode with auto-reload
- **background** (`bool`): Run servers in background
- **ui** (`bool`): Also start UI development server
- **sample_data** (`bool`): Create sample data before starting
- **kill_on_exit** (`bool`): **NEW** - If `True` and `background=True`, background processes will be automatically killed when the main process exits

#### Returns

Dictionary with PIDs: `{"api": int, "ui": int}` (ui only if requested)

## Implementation Details

### Process Management

When `kill_on_exit=True`:

1. **Process Group**: Subprocess stays in parent's process group (no `start_new_session`)
2. **PID Tracking**: Background PIDs are tracked in `_background_pids` list
3. **Cleanup Registration**: 
   - `atexit` handler registered for normal exits
   - Signal handlers for `SIGTERM` and `SIGINT`
4. **Graceful Shutdown**: Attempts `SIGTERM` first, then `SIGKILL` if needed

### Signal Handling

The cleanup handler responds to:
- Normal script exit (via `atexit`)
- `SIGTERM` (termination signal)
- `SIGINT` (Ctrl+C)

## Use Cases

### 1. Production Deployment (Default)

```python
# Start server that persists after deployment script exits
manager = ServerManager()
manager.start(background=True, kill_on_exit=False)
```

### 2. Testing & Development

```python
# Server automatically cleaned up after tests
manager = ServerManager()
manager.start(background=True, kill_on_exit=True)
# Run tests...
# Server killed automatically when script ends
```

### 3. Temporary Services

```python
# Start temporary server for a specific task
with TemporaryServer():
    manager = ServerManager()
    manager.start(background=True, kill_on_exit=True)
    # Do work...
# Server cleaned up automatically
```

## Examples

See [`examples/server_lifecycle_demo.py`](../examples/server_lifecycle_demo.py) for complete demonstrations:

```bash
# Demo default behavior
python examples/server_lifecycle_demo.py --mode default

# Demo kill-on-exit behavior
python examples/server_lifecycle_demo.py --mode kill-on-exit

# Quick test
python examples/server_lifecycle_demo.py --mode test-exit
```

## Troubleshooting

### Process Still Running After Exit

Check if `kill_on_exit` was enabled:

```bash
# Check if server is running
ps aux | grep uvicorn

# Stop manually if needed
python -c "from molexp.server import ServerManager; ServerManager().stop()"
```

### Processes Not Being Killed

Possible causes:
1. Signal handlers blocked or overridden
2. Process became orphaned (check process group)
3. Hard crash preventing cleanup (use `kill -9` as last resort)

Debug with:
```bash
# Check process group
ps -o pid,pgid,command | grep uvicorn
```

## Best Practices

1. **Production**: Use `kill_on_exit=False` (default) for persistent services
2. **Testing**: Use `kill_on_exit=True` to avoid orphaned processes
3. **Development**: Consider foreground mode (`background=False`) for easier debugging
4. **Monitoring**: Always check server status after starting:
   ```python
   if not manager.is_running():
       print("Server failed to start!")
   ```
