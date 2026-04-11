"""Server lifecycle management for molexp API server."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator

try:
    import psutil
except ImportError:
    psutil = None


class ServerManager:
    """Manage molexp API server lifecycle."""

    def __init__(self, config_dir: Path | None = None) -> None:
        """Initialize server manager.

        Args:
            config_dir: Configuration directory (default: ~/.molexp)
        """
        self.config_dir = config_dir or Path.home() / ".molexp"
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.pid_file = self.config_dir / "server.pid"
        self.ui_pid_file = self.config_dir / "ui.pid"
        self.log_dir = self.config_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)

        self.server_log = self.log_dir / "server.log"
        self.ui_log = self.log_dir / "ui.log"
        
        # Track background processes for cleanup
        self._background_pids: list[int] = []

    def start(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        dev: bool = True,
        background: bool = False,
        ui: bool = False,
        sample_data: bool = False,
        kill_on_exit: bool = False,
    ) -> dict[str, int]:
        """Start API server and optionally UI server.

        Args:
            host: Host address
            port: Port number
            dev: Development mode with auto-reload
            background: Run in background (daemon mode)
            ui: Also start UI dev server
            sample_data: Create sample data before starting
            kill_on_exit: If True and background=True, kill background processes when main process exits

        Returns:
            Dictionary with 'api' and optionally 'ui' PIDs

        Raises:
            RuntimeError: If server is already running or fails to start
        """
        if self.is_running():
            raise RuntimeError(
                f"Server is already running (PID: {self._read_pid(self.pid_file)})"
            )

        pids = {}

        # Create sample data if requested
        if sample_data:
            self._create_sample_data()

        # Start API server
        api_pid = self._start_api_server(host, port, dev, background, kill_on_exit)
        pids["api"] = api_pid
        self._write_pid(self.pid_file, api_pid)
        
        # Track for cleanup if needed
        if background and kill_on_exit:
            self._background_pids.append(api_pid)

        # Wait for API to be healthy
        if not self._wait_for_health(host, port):
            self.stop()
            raise RuntimeError("API server failed to start. Check logs for details.")

        # Start UI server if requested
        if ui:
            ui_pid = self._start_ui_server(background, kill_on_exit)
            pids["ui"] = ui_pid
            self._write_pid(self.ui_pid_file, ui_pid)
            
            # Track for cleanup if needed
            if background and kill_on_exit:
                self._background_pids.append(ui_pid)
        
        # Register cleanup handler if kill_on_exit is True
        if background and kill_on_exit:
            self._register_cleanup_handler()

        return pids

    def stop(self, ui: bool = False, timeout: int = 10) -> bool:
        """Stop running server(s).

        Args:
            ui: Also stop UI server if running
            timeout: Seconds to wait for graceful shutdown

        Returns:
            True if stopped successfully, False otherwise
        """
        success = True

        # Stop API server
        if self.pid_file.exists():
            pid = self._read_pid(self.pid_file)
            if pid and self._is_process_running(pid):
                success &= self._stop_process(pid, timeout)
            self.pid_file.unlink(missing_ok=True)

        # Stop UI server if requested
        if ui and self.ui_pid_file.exists():
            pid = self._read_pid(self.ui_pid_file)
            if pid and self._is_process_running(pid):
                success &= self._stop_process(pid, timeout)
            self.ui_pid_file.unlink(missing_ok=True)

        return success

    def status(self) -> dict:
        """Get server status information.

        Returns:
            Dictionary with status information
        """
        api_status = self._get_process_status(self.pid_file, "API")
        ui_status = self._get_process_status(self.ui_pid_file, "UI")

        return {
            "api": api_status,
            "ui": ui_status,
        }

    def is_running(self) -> bool:
        """Check if API server is running.

        Returns:
            True if running, False otherwise
        """
        if not self.pid_file.exists():
            return False

        pid = self._read_pid(self.pid_file)
        return pid is not None and self._is_process_running(pid)

    def get_logs(
        self, lines: int = 50, follow: bool = False, ui: bool = False
    ) -> Iterator[str]:
        """Get server logs.

        Args:
            lines: Number of lines to show
            follow: Follow log output (tail -f style)
            ui: Show UI logs instead of API logs

        Yields:
            Log lines
        """
        log_file = self.ui_log if ui else self.server_log

        if not log_file.exists():
            yield f"Log file not found: {log_file}"
            return

        if follow:
            # Follow mode - tail -f style
            with open(log_file, "r") as f:
                # Seek to end minus N lines
                f.seek(0, 2)  # Go to end
                file_size = f.tell()

                # Read last N lines
                block_size = 1024
                blocks = []
                num_lines = 0

                while file_size > 0 and num_lines < lines:
                    read_size = min(block_size, file_size)
                    f.seek(file_size - read_size)
                    block = f.read(read_size)
                    blocks.append(block)
                    num_lines += block.count("\n")
                    file_size -= read_size

                # Yield initial lines
                content = "".join(reversed(blocks))
                initial_lines = content.split("\n")[-lines:]
                for line in initial_lines:
                    if line:
                        yield line

                # Follow new lines
                f.seek(0, 2)  # Go to end
                while True:
                    line = f.readline()
                    if line:
                        yield line.rstrip()
                    else:
                        time.sleep(0.1)
        else:
            # Just read last N lines
            with open(log_file, "r") as f:
                all_lines = f.readlines()
                for line in all_lines[-lines:]:
                    yield line.rstrip()

    # ============ Private Methods ============

    def _start_api_server(
        self, host: str, port: int, dev: bool, background: bool, kill_on_exit: bool = False
    ) -> int:
        """Start API server process.
        
        Args:
            host: Host address
            port: Port number
            dev: Development mode with auto-reload
            background: Run in background
            kill_on_exit: If True and background=True, process will be killed when parent exits
        
        Returns:
            Process PID
        """
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "molexp.server.app:app",
            "--host",
            host,
            "--port",
            str(port),
        ]

        if dev:
            cmd.append("--reload")

        if background:
            # Run in background
            with open(self.server_log, "a") as log:
                # Only detach from parent if kill_on_exit is False
                # When kill_on_exit is True, keep it in the same process group
                process = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=(not kill_on_exit),
                )
            return process.pid
        else:
            # Run in foreground
            process = subprocess.Popen(cmd)
            return process.pid

    def _start_ui_server(self, background: bool, kill_on_exit: bool = False) -> int:
        """Start UI dev server process.
        
        Args:
            background: Run in background
            kill_on_exit: If True and background=True, process will be killed when parent exits
        
        Returns:
            Process PID
        """
        ui_dir = Path(__file__).parent.parent.parent / "ui"

        if not ui_dir.exists():
            raise RuntimeError(f"UI directory not found: {ui_dir}")

        cmd = ["npm", "run", "dev"]

        if background:
            with open(self.ui_log, "a") as log:
                # Only detach from parent if kill_on_exit is False
                # When kill_on_exit is True, keep it in the same process group
                process = subprocess.Popen(
                    cmd,
                    cwd=ui_dir,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=(not kill_on_exit),
                )
            return process.pid
        else:
            process = subprocess.Popen(cmd, cwd=ui_dir)
            return process.pid

    def _create_sample_data(self) -> None:
        """Create sample data using create_sample_data.py."""
        script_path = Path(__file__).parent.parent.parent / "create_sample_data.py"

        if not script_path.exists():
            raise RuntimeError(f"Sample data script not found: {script_path}")

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to create sample data: {result.stderr}")

    def _wait_for_health(self, host: str, port: int, max_retries: int = 30) -> bool:
        """Wait for server to be healthy."""
        url = f"http://{host}:{port}/health"

        for _ in range(max_retries):
            try:
                with urllib.request.urlopen(url, timeout=1) as response:
                    if response.status == 200:
                        return True
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(1)

        return False

    def _stop_process(self, pid: int, timeout: int) -> bool:
        """Stop a process gracefully."""
        if not psutil:
            # Fallback to simple signal-based approach
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(1)

                # Check if still running
                try:
                    os.kill(pid, 0)  # Check if process exists
                    # Still running, force kill
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    # Process terminated
                    pass
                return True
            except ProcessLookupError:
                return True
            except Exception:
                return False

        # Use psutil for better process management
        try:
            process = psutil.Process(pid)
            process.terminate()

            try:
                process.wait(timeout=timeout)
            except psutil.TimeoutExpired:
                # Force kill
                process.kill()
                process.wait(timeout=5)

            return True
        except psutil.NoSuchProcess:
            return True
        except Exception:
            return False

    def _is_process_running(self, pid: int) -> bool:
        """Check if process is running."""
        if psutil:
            return psutil.pid_exists(pid)

        # Fallback
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except Exception:
            return False

    def _get_process_status(self, pid_file: Path, name: str) -> dict:
        """Get status of a process."""
        if not pid_file.exists():
            return {
                "name": name,
                "running": False,
                "pid": None,
            }

        pid = self._read_pid(pid_file)
        if not pid or not self._is_process_running(pid):
            return {
                "name": name,
                "running": False,
                "pid": pid,
            }

        status = {
            "name": name,
            "running": True,
            "pid": pid,
        }

        if psutil:
            try:
                process = psutil.Process(pid)
                status["uptime"] = time.time() - process.create_time()
                status["memory_mb"] = process.memory_info().rss / 1024 / 1024
                status["cpu_percent"] = process.cpu_percent(interval=0.1)
            except psutil.NoSuchProcess:
                status["running"] = False

        return status

    def _read_pid(self, pid_file: Path) -> int | None:
        """Read PID from file."""
        try:
            return int(pid_file.read_text().strip())
        except (ValueError, FileNotFoundError):
            return None

    def _write_pid(self, pid_file: Path, pid: int) -> None:
        """Write PID to file."""
        pid_file.write_text(str(pid))
    
    def _register_cleanup_handler(self) -> None:
        """Register cleanup handler to kill background processes on exit."""
        import atexit
        
        # Register cleanup function
        atexit.register(self._cleanup_background_processes)
        
        # Also handle signals for graceful shutdown
        def signal_handler(signum, frame):
            self._cleanup_background_processes()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def _cleanup_background_processes(self) -> None:
        """Clean up tracked background processes."""
        for pid in self._background_pids:
            if self._is_process_running(pid):
                try:
                    self._stop_process(pid, timeout=5)
                except Exception:
                    pass
        self._background_pids.clear()
