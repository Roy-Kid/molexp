"""Codex app-server JSON-RPC adapter.

Implements :class:`molexp.plugins.coding_agent.CodingAgentClient` by spawning
``codex app-server`` as a long-lived subprocess and exchanging JSON-RPC frames
over stdio. One client instance manages one app-server process.

Reverse-RPC tool dispatch (the Codex agent calling out to the host for
``item/tool/call``) is forwarded to a caller-supplied :class:`ToolHandler`.
The plugin itself ships zero builtin tools — symphony (or any other host)
attaches its tool handlers when constructing the client.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from molexp.plugins.agent_codex.config import CodexConfig
from molexp.plugins.coding_agent import (
    AgentError,
    AgentEventCallback,
    AgentTurnInputRequiredError,
    TurnResult,
    drain_stderr,
    emit_event,
    terminate_subprocess,
)


@runtime_checkable
class ToolHandler(Protocol):
    """Callable surface for reverse-RPC tool dispatch from Codex.

    Implementations receive the ``params`` dict of an ``item/tool/call``
    JSON-RPC request and return the JSON-RPC ``result`` dict the
    app-server expects (``{"success": bool, "contentItems": [...]}``).

    The handler is invoked **synchronously inside the JSON-RPC reader
    loop**; long-running work should not be done here or it stalls the
    reader and blocks every other in-flight request. Defer slow I/O to
    a background task and return a placeholder result if necessary.
    """

    def execute_tool(self, params: dict[str, Any] | None) -> dict[str, Any]: ...


class CodexAppServerClient:
    """``CodingAgentClient`` implementation backed by the Codex app-server.

    Args:
        config: Provider configuration; see :class:`CodexConfig`.
        workspace: Working directory the app-server runs in.
        on_event: Sink for normalized events (sync or async).
        tool_handler: Optional handler invoked when the Codex agent calls
            a host-provided tool via reverse-RPC. ``None`` makes any tool
            call return an "unsupported_tool_call" failure response.
    """

    def __init__(
        self,
        config: CodexConfig,
        workspace: Path,
        on_event: AgentEventCallback,
        tool_handler: ToolHandler | None = None,
    ) -> None:
        self.config = config
        self.workspace = workspace
        self.on_event = on_event
        self.tool_handler = tool_handler
        self.process: asyncio.subprocess.Process | None = None
        self._next_id = 1
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        # Turn completion arrives as a notification, not a JSON-RPC reply,
        # so it cannot share `_pending`.
        self._active_turn: asyncio.Future[dict[str, Any]] | None = None

    @property
    def pid(self) -> int | None:
        return self.process.pid if self.process else None

    async def start(self) -> None:
        self.process = await asyncio.create_subprocess_exec(
            "bash",
            "-lc",
            self.config.command,
            cwd=str(self.workspace),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._reader_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(drain_stderr(self.process, self.on_event))
        await self.request(
            "initialize",
            {
                "clientInfo": {
                    "name": "molexp.plugins.agent_codex",
                    "title": "MolExp Codex Plugin",
                    "version": "0.1.0",
                },
                "capabilities": {"experimentalApi": True},
            },
            timeout_ms=self.config.read_timeout_ms,
        )
        await emit_event(
            self.on_event, {"event": "app_server_started", "codex_app_server_pid": self.pid}
        )

    async def start_thread(self) -> str:
        params: dict[str, Any] = {
            "cwd": str(self.workspace),
            "serviceName": "molexp.plugins.agent_codex",
            "ephemeral": False,
        }
        if self.config.approval_policy is not None:
            params["approvalPolicy"] = self.config.approval_policy
        if self.config.thread_sandbox is not None:
            params["sandbox"] = self.config.thread_sandbox
        if self.config.model is not None:
            params["model"] = self.config.model
        response = await self.request(
            "thread/start", params, timeout_ms=self.config.read_timeout_ms
        )
        thread = response.get("thread") or {}
        thread_id = thread.get("id")
        if not thread_id:
            raise AgentError("response_error missing thread id")
        await emit_event(
            self.on_event,
            {
                "event": "thread_started",
                "thread_id": thread_id,
                "codex_app_server_pid": self.pid,
            },
        )
        return str(thread_id)

    async def run_turn(self, thread_id: str, prompt: str) -> TurnResult:
        params: dict[str, Any] = {
            "threadId": thread_id,
            "cwd": str(self.workspace),
            "input": [{"type": "text", "text": prompt}],
        }
        if self.config.approval_policy is not None:
            params["approvalPolicy"] = self.config.approval_policy
        if self.config.turn_sandbox_policy is not None:
            params["sandboxPolicy"] = self.config.turn_sandbox_policy
        if self.config.model is not None:
            params["model"] = self.config.model
        self._active_turn = asyncio.get_running_loop().create_future()
        response = await self.request(
            "turn/start", params, timeout_ms=self.config.read_timeout_ms
        )
        turn = response.get("turn") or {}
        turn_id = turn.get("id")
        if not turn_id:
            raise AgentError("response_error missing turn id")
        await emit_event(
            self.on_event,
            {"event": "turn_started", "thread_id": thread_id, "turn_id": turn_id},
        )
        try:
            completed = await asyncio.wait_for(
                self._active_turn,
                timeout=self.config.turn_timeout_ms / 1000,
            )
        except asyncio.TimeoutError as exc:
            raise AgentError("turn_timeout") from exc
        status = str(((completed.get("params") or {}).get("turn") or {}).get("status", "completed"))
        if status != "completed":
            raise AgentError(f"turn_failed status={status}")
        return TurnResult(thread_id=thread_id, turn_id=str(turn_id), status=status)

    async def request(self, method: str, params: dict[str, Any], timeout_ms: int) -> dict[str, Any]:
        if self.process is None or self.process.stdin is None:
            raise AgentError("port_exit app-server is not running")
        request_id = self._next_id
        self._next_id += 1
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        await self._write({"id": request_id, "method": method, "params": params})
        try:
            return await asyncio.wait_for(future, timeout=timeout_ms / 1000)
        except asyncio.TimeoutError as exc:
            self._pending.pop(request_id, None)
            raise AgentError(f"response_timeout method={method}") from exc

    async def close(self) -> None:
        # Unblock anyone awaiting an RPC reply or turn completion before the
        # subprocess goes away; otherwise they hang until their own timeout.
        for future in self._pending.values():
            if not future.done():
                future.set_exception(AgentError("port_exit"))
        self._pending.clear()
        if self._active_turn is not None and not self._active_turn.done():
            self._active_turn.set_exception(AgentError("port_exit"))
        if self._reader_task is not None:
            self._reader_task.cancel()
        if self._stderr_task is not None:
            self._stderr_task.cancel()
        await terminate_subprocess(self.process)
        self.process = None

    # ── internal ────────────────────────────────────────────────────────

    async def _write(self, payload: dict[str, Any]) -> None:
        assert self.process is not None and self.process.stdin is not None
        self.process.stdin.write(json.dumps(payload).encode("utf-8") + b"\n")
        await self.process.stdin.drain()

    async def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        while True:
            line = await self.process.stdout.readline()
            if not line:
                for future in self._pending.values():
                    if not future.done():
                        future.set_exception(AgentError("port_exit"))
                return
            try:
                message = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                await emit_event(
                    self.on_event,
                    {"event": "malformed", "message": line[:200].decode("utf-8", "replace")},
                )
                continue
            await self._handle_message(message)

    async def _handle_message(self, message: dict[str, Any]) -> None:
        if "id" in message and ("result" in message or "error" in message):
            future = self._pending.pop(int(message["id"]), None)
            if future is None:
                return
            if "error" in message:
                future.set_exception(AgentError(f"response_error {message['error']}"))
            else:
                future.set_result(message.get("result") or {})
            return
        if "id" in message and "method" in message:
            await self._handle_server_request(message)
            return
        method = message.get("method")
        params = message.get("params") or {}
        await emit_event(self.on_event, {"event": method or "other_message", "payload": params})
        if method == "thread/tokenUsage/updated":
            await emit_event(
                self.on_event, {"event": "token_usage", "usage": params.get("tokenUsage")}
            )
        elif method == "account/rateLimits/updated":
            await emit_event(
                self.on_event, {"event": "rate_limits", "rateLimits": params.get("rateLimits")}
            )
        elif method == "turn/completed":
            if self._active_turn and not self._active_turn.done():
                self._active_turn.set_result(message)

    async def _handle_server_request(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        request_id = message["id"]
        params = message.get("params") or {}
        if method in (
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
            "item/permissions/requestApproval",
        ):
            await emit_event(self.on_event, {"event": "approval_auto_approved", "payload": params})
            await self._write({"id": request_id, "result": {"decision": "acceptForSession"}})
        elif method == "item/tool/requestUserInput":
            await self._write(
                {
                    "id": request_id,
                    "error": {"code": -32000, "message": "user input is not available"},
                }
            )
            if self._active_turn and not self._active_turn.done():
                self._active_turn.set_exception(AgentTurnInputRequiredError("turn_input_required"))
        elif method == "item/tool/call":
            result = self._dispatch_tool(params)
            await self._write({"id": request_id, "result": result})
        else:
            await self._write(
                {
                    "id": request_id,
                    "error": {"code": -32601, "message": f"unsupported request {method}"},
                }
            )

    def _dispatch_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        if self.tool_handler is None:
            return _unsupported_tool_response(params.get("tool"))
        try:
            return self.tool_handler.execute_tool(params)
        except Exception as exc:  # noqa: BLE001 — surface as JSON-RPC failure
            return {
                "success": False,
                "contentItems": [
                    {
                        "type": "inputText",
                        "text": json.dumps(
                            {"error": "tool_handler_failure", "detail": str(exc)}
                        ),
                    }
                ],
            }

def _unsupported_tool_response(tool: Any) -> dict[str, Any]:
    return {
        "success": False,
        "contentItems": [
            {
                "type": "inputText",
                "text": json.dumps({"error": "unsupported_tool_call", "tool": tool}),
            }
        ],
    }
