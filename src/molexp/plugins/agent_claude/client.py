"""Claude Code CLI subprocess adapter.

Implements :class:`molexp.plugins.coding_agent.CodingAgentClient` by spawning
``claude -p`` once per turn, piping the rendered prompt through stdin, and
parsing the ``--output-format stream-json`` event stream into normalized
events plus a single :class:`TurnResult` per turn.

The plugin is intentionally focused: it owns subprocess lifecycle, JSON
event parsing, and env-var hygiene. It does **not** assemble MCP server
configurations — callers pre-render their own ``mcp.json`` (if any) and
pass its path via :class:`ClaudeCliConfig.mcp_config`.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Literal

from molexp.plugins.agent_claude.config import ClaudeCliConfig, SubagentDef
from molexp.plugins.coding_agent import (
    AgentError,
    AgentEventCallback,
    TurnResult,
    drain_stderr,
    emit_event,
    terminate_subprocess,
)

ENV_VARS_TO_STRIP_FOR_SUBSCRIPTION = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
)


class ClaudeCliClient:
    """``CodingAgentClient`` implementation backed by the Claude Code CLI.

    Args:
        config: Provider configuration; see :class:`ClaudeCliConfig`.
        workspace: Working directory the CLI runs in. Files written by the
            agent land here.
        on_event: Callback invoked with each normalized event dict. May be
            sync or async (return value awaited if a coroutine).
        models: Optional tier → concrete-model map. Used to resolve the
            main agent's ``--model`` value (via
            :attr:`ClaudeCliConfig.model_tier`) and per-sub-agent models.
        subagents: Sub-agent definitions forwarded to the CLI's
            ``--agents`` flag.
    """

    def __init__(
        self,
        config: ClaudeCliConfig,
        workspace: Path,
        on_event: AgentEventCallback,
        models: dict[str, str] | None = None,
        subagents: tuple[SubagentDef, ...] = (),
    ) -> None:
        self.config = config
        self.workspace = workspace
        self.on_event = on_event
        self.models: dict[str, str] = dict(models or {})
        self.subagents: tuple[SubagentDef, ...] = subagents
        self._session_uuid: str | None = None
        self._first_turn_started = False
        self._current_proc: asyncio.subprocess.Process | None = None
        # Claude CLI reports per-turn usage; orchestrator expects cumulative
        # totals per thread. Accumulate here so emitted token_usage events
        # grow monotonically across turns.
        self._cumulative_input_tokens = 0
        self._cumulative_output_tokens = 0

    @property
    def pid(self) -> int | None:
        proc = self._current_proc
        if proc is not None and proc.returncode is None:
            return proc.pid
        return None

    async def start(self) -> None:
        if shutil.which(self.config.command) is None:
            raise AgentError(f"claude_cli_not_found command={self.config.command!r}")
        await emit_event(
            self.on_event,
            {
                "event": "claude_cli_ready",
                "command": self.config.command,
                "mcp_config": str(self.config.mcp_config) if self.config.mcp_config else None,
            },
        )

    async def start_thread(self) -> str:
        thread_id = str(uuid.uuid4())
        self._session_uuid = thread_id
        self._first_turn_started = False
        await emit_event(self.on_event, {"event": "thread_started", "thread_id": thread_id})
        return thread_id

    async def run_turn(self, thread_id: str, prompt: str) -> TurnResult:
        if self._session_uuid is None or thread_id != self._session_uuid:
            raise AgentError(
                f"thread_id_mismatch expected={self._session_uuid!r} got={thread_id!r}"
            )
        cmd = self._build_command(first_turn=not self._first_turn_started)
        env = self._prepare_env()
        turn_id = str(uuid.uuid4())
        await emit_event(
            self.on_event,
            {"event": "turn_started", "thread_id": thread_id, "turn_id": turn_id},
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.workspace),
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise AgentError(f"claude_cli_not_found command={self.config.command!r}") from exc
        self._current_proc = proc
        await emit_event(self.on_event, {"event": "subprocess_started", "pid": proc.pid})

        assert proc.stdin is not None
        try:
            proc.stdin.write(prompt.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise AgentError(f"claude_cli_subprocess_failure stdin={exc}") from exc

        stderr_task = asyncio.create_task(drain_stderr(proc, self.on_event))
        result_event: dict[str, Any] | None = None
        first_event_seen = False
        read_timeout = self.config.read_timeout_ms / 1000
        turn_timeout = self.config.turn_timeout_ms / 1000
        assert proc.stdout is not None
        try:
            while True:
                # asyncio.wait_for cancels the awaiter on timeout, not the
                # underlying transport read; we kill the subprocess to make
                # the buffer ownership unambiguous.
                timeout = read_timeout if not first_event_seen else turn_timeout
                try:
                    line = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout)
                except asyncio.TimeoutError as exc:
                    proc.kill()
                    code = "response_timeout" if not first_event_seen else "turn_timeout"
                    raise AgentError(code) from exc
                if not line:
                    break
                first_event_seen = True
                try:
                    message = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    await emit_event(
                        self.on_event,
                        {"event": "malformed", "message": line[:200].decode("utf-8", "replace")},
                    )
                    continue
                result = await self._dispatch_message(message, thread_id, turn_id)
                if result is not None:
                    result_event = result
        finally:
            # Stream EOF reached naturally → process is exiting; just wait.
            # Kill is reserved for the timeout path above.
            try:
                rc = await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
                rc = await proc.wait()
            self._current_proc = None
            stderr_task.cancel()

        if result_event is None:
            raise AgentError(f"claude_cli_subprocess_failure exit={rc} no_result_event=true")

        is_error = bool(result_event.get("is_error"))
        subtype = str(result_event.get("subtype") or "")
        terminal_reason = str(result_event.get("terminal_reason") or "")
        status: Literal["completed", "failed"] = (
            "completed" if not is_error and subtype == "success" else "failed"
        )

        if status != "completed":
            stop_reason = str(result_event.get("stop_reason") or "")
            await emit_event(
                self.on_event,
                {
                    "event": "turn_failed",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "stop_reason": stop_reason,
                    "terminal_reason": terminal_reason,
                    "api_error_status": result_event.get("api_error_status"),
                },
            )
            raise AgentError(
                f"turn_failed subtype={subtype!r} stop_reason={stop_reason!r} "
                f"terminal_reason={terminal_reason!r}"
            )

        await emit_event(
            self.on_event,
            {
                "event": "turn_completed",
                "thread_id": thread_id,
                "turn_id": turn_id,
                "stop_reason": result_event.get("stop_reason"),
            },
        )
        self._first_turn_started = True
        return TurnResult(thread_id=thread_id, turn_id=turn_id, status=status)

    async def close(self) -> None:
        proc = self._current_proc
        self._current_proc = None
        await terminate_subprocess(proc)

    def _build_command(self, *, first_turn: bool) -> list[str]:
        cmd: list[str] = [
            self.config.command,
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--permission-mode",
            self.config.permission_mode,
        ]
        if first_turn:
            cmd += ["--session-id", self._require_uuid()]
        else:
            cmd += ["--resume", self._require_uuid()]
        main_model = self._resolve_main_model()
        if main_model:
            cmd += ["--model", main_model]
        if self.config.mcp_config is not None:
            cmd += ["--mcp-config", str(self.config.mcp_config)]
            if self.config.strict_mcp_config:
                cmd.append("--strict-mcp-config")
        agents_payload = self._build_agents_payload()
        if agents_payload:
            cmd += ["--agents", json.dumps(agents_payload, sort_keys=True)]
        cmd.extend(self.config.extra_args)
        return cmd

    def _resolve_main_model(self) -> str | None:
        """Pick the model id for the *main* agent.

        Resolution order:
            1. ``ClaudeCliConfig.model`` — explicit override always wins.
            2. ``models[ClaudeCliConfig.model_tier]`` — tier resolution.
            3. ``None`` — let the CLI pick its built-in default.
        """
        if self.config.model:
            return self.config.model
        if self.models and self.config.model_tier in self.models:
            return self.models[self.config.model_tier]
        return None

    def _build_agents_payload(self) -> dict[str, dict[str, Any]] | None:
        """Render ``subagents`` into the JSON shape ``claude --agents`` expects."""
        if not self.subagents:
            return None
        payload: dict[str, dict[str, Any]] = {}
        for sub in self.subagents:
            entry: dict[str, Any] = {
                "description": sub.description,
                "prompt": sub.prompt,
            }
            model = self.models.get(sub.tier)
            if model:
                entry["model"] = model
            payload[sub.name] = entry
        return payload

    def _require_uuid(self) -> str:
        if self._session_uuid is None:
            raise AgentError("session_uuid_missing")
        return self._session_uuid

    def _prepare_env(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.config.strip_anthropic_api_key_env:
            for key in ENV_VARS_TO_STRIP_FOR_SUBSCRIPTION:
                env.pop(key, None)
        return env

    async def _dispatch_message(
        self,
        message: dict[str, Any],
        thread_id: str,
        turn_id: str,
    ) -> dict[str, Any] | None:
        """Translate one stream-json message to events; return the ``result``
        message when the turn completes, otherwise ``None``."""
        msg_type = message.get("type")
        subtype = message.get("subtype")
        if msg_type == "system" and subtype == "init":
            await emit_event(
                self.on_event,
                {
                    "event": "system_init",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "claude_code_version": message.get("claude_code_version"),
                    "model": message.get("model"),
                    "permission_mode": message.get("permissionMode"),
                    "api_key_source": message.get("apiKeySource"),
                    "mcp_servers": message.get("mcp_servers"),
                    "session_id": message.get("session_id"),
                },
            )
            return None
        if msg_type == "rate_limit_event":
            info = message.get("rate_limit_info") or {}
            await emit_event(self.on_event, {"event": "rate_limits", "rateLimits": info})
            status = info.get("status")
            if status and status not in ("allowed", "ok"):
                await emit_event(
                    self.on_event,
                    {
                        "event": "rate_limit_rejected",
                        "rateLimits": info,
                        "thread_id": thread_id,
                        "turn_id": turn_id,
                        "resetsAt": info.get("resetsAt"),
                    },
                )
            return None
        if msg_type == "assistant":
            for tool in _extract_tool_uses(message):
                tool_name = tool.get("name") or ""
                tool_input = tool.get("input") if isinstance(tool.get("input"), dict) else {}
                payload: dict[str, Any] = {
                    "event": "tool_use",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "tool_name": tool_name,
                    "tool_use_id": tool.get("id"),
                    "tool_summary": _summarize_tool_input(tool_name, tool_input),
                }
                if tool_name == "TodoWrite":
                    payload["todos"] = _normalize_todos(tool_input.get("todos"))
                await emit_event(self.on_event, payload)
            assistant_text = _extract_assistant_text(message)
            if assistant_text:
                await emit_event(
                    self.on_event,
                    {
                        "event": "assistant_text",
                        "thread_id": thread_id,
                        "turn_id": turn_id,
                        "text": assistant_text,
                    },
                )
            return None
        if msg_type == "user":
            return None
        if msg_type == "result":
            usage = _normalize_usage(message.get("usage") or {})
            if usage:
                self._cumulative_input_tokens += int(usage.get("inputTokens", 0))
                self._cumulative_output_tokens += int(usage.get("outputTokens", 0))
                cumulative = {
                    "inputTokens": self._cumulative_input_tokens,
                    "outputTokens": self._cumulative_output_tokens,
                    "totalTokens": self._cumulative_input_tokens + self._cumulative_output_tokens,
                }
                await emit_event(
                    self.on_event, {"event": "token_usage", "usage": {"total": cumulative}}
                )
            await emit_event(
                self.on_event,
                {
                    "event": "result",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "subtype": subtype,
                    "is_error": message.get("is_error"),
                    "stop_reason": message.get("stop_reason"),
                    "duration_ms": message.get("duration_ms"),
                    "total_cost_usd": message.get("total_cost_usd"),
                    "permission_denials": message.get("permission_denials") or [],
                },
            )
            return message
        # Categorize less-common events so long pauses are not opaque.
        category_type = str(msg_type or "unknown").replace("/", "_")
        category_sub = str(subtype or "").replace("/", "_") if subtype else ""
        category = "_".join(part for part in ("claude", category_type, category_sub) if part)
        await emit_event(
            self.on_event,
            {
                "event": category,
                "payload_type": msg_type,
                "payload_subtype": subtype,
                "payload": message,
            },
        )
        return None


# ── pure helpers (kept module-private so the plugin surface stays minimal) ──


def _extract_tool_uses(message: dict[str, Any]) -> list[dict[str, Any]]:
    inner = message.get("message")
    if not isinstance(inner, dict):
        return []
    content = inner.get("content")
    if not isinstance(content, list):
        return []
    return [
        block
        for block in content
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]


def _extract_assistant_text(message: dict[str, Any]) -> str:
    inner = message.get("message")
    if not isinstance(inner, dict):
        return ""
    content = inner.get("content")
    if not isinstance(content, list):
        return ""
    parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    text = "\n".join(part for part in parts if part).strip()
    return text[:500]


def _summarize_tool_input(name: str, input_data: dict[str, Any]) -> str:
    """Best-effort one-line summary of a Claude tool invocation for human logs."""
    if not isinstance(input_data, dict):
        return name
    if name == "Bash":
        cmd = str(input_data.get("command", ""))
        return _short(cmd, limit=120)
    if name == "Read":
        return str(input_data.get("file_path", ""))
    if name == "Write":
        return str(input_data.get("file_path", ""))
    if name == "Edit":
        path = str(input_data.get("file_path", ""))
        new = _short(str(input_data.get("new_string", "")), limit=60)
        return f"{path} -> {new}" if new else path
    if name == "Glob":
        return f"{input_data.get('pattern', '')} in {input_data.get('path', '.')}"
    if name == "Grep":
        return f"{input_data.get('pattern', '')} in {input_data.get('path', '.')}"
    if name == "TodoWrite":
        todos = input_data.get("todos") or []
        if not isinstance(todos, list):
            return "TodoWrite"
        done = sum(1 for t in todos if isinstance(t, dict) and t.get("status") == "completed")
        return f"{len(todos)} items ({done} done)"
    if name == "WebFetch":
        return str(input_data.get("url", ""))
    if name == "WebSearch":
        return str(input_data.get("query", ""))
    if name.startswith("mcp__"):
        return _short(json.dumps(input_data, sort_keys=True), limit=120)
    return _short(json.dumps(input_data, sort_keys=True), limit=80)


def _normalize_todos(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    todos: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        todos.append(
            {
                "content": str(item.get("content", ""))[:200],
                "status": str(item.get("status", "")),
                "active_form": str(item.get("activeForm", item.get("active_form", "")))[:200],
            }
        )
    return todos


def _short(value: str, *, limit: int) -> str:
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def _normalize_usage(usage: dict[str, Any]) -> dict[str, int]:
    """Map Claude CLI usage shape into a normalized {inputTokens, outputTokens, totalTokens} dict."""
    if not isinstance(usage, dict):
        return {}
    base_input = int(usage.get("input_tokens") or 0)
    cache_creation = int(usage.get("cache_creation_input_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    total_input = base_input + cache_creation + cache_read
    return {
        "inputTokens": total_input,
        "outputTokens": output_tokens,
        "totalTokens": total_input + output_tokens,
    }
