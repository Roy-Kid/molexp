import { describe, expect, it } from "@rstest/core";

import type { ApiSessionEvent } from "@/app/types";
import { appendCoalescedEvent, coalesceStreamEvents, foldStreamedTurn } from "../agentEvents";

const ev = (type: string, payload: Record<string, unknown> = {}): ApiSessionEvent => ({
  type,
  ts: "2026-05-30T00:00:00Z",
  payload,
});

describe("coalesceStreamEvents", () => {
  it("merges consecutive thinking and token deltas", () => {
    const merged = coalesceStreamEvents([
      ev("thinking_delta", { text: "a" }),
      ev("thinking_delta", { text: "b" }),
      ev("tool_call_started", { tool_name: "x" }),
      ev("token_delta", { text: "1" }),
      ev("token_delta", { text: "2" }),
    ]);
    expect(merged).toHaveLength(3);
    expect(merged[0]).toMatchObject({ type: "thinking_delta", payload: { text: "ab" } });
    expect(merged[1].type).toBe("tool_call_started");
    expect(merged[2]).toMatchObject({ type: "token_delta", payload: { text: "12" } });
  });

  it("appendCoalescedEvent extends the last delta in place", () => {
    let rows = [ev("thinking_delta", { text: "hi" })];
    rows = appendCoalescedEvent(rows, ev("thinking_delta", { text: "!" }));
    expect(rows).toHaveLength(1);
    expect(rows[0].payload).toMatchObject({ text: "hi!" });
  });
});

describe("foldStreamedTurn", () => {
  it("concatenates token/thinking deltas separately and completes one tool call", () => {
    const turn = foldStreamedTurn([
      ev("thinking_delta", { text: "let me " }),
      ev("thinking_delta", { text: "reason" }),
      ev("token_delta", { text: "the " }),
      ev("token_delta", { text: "answer" }),
      ev("tool_call_started", { tool_name: "read_file", args_summary: "path=a.py" }),
      ev("tool_call_completed", { tool_name: "read_file", result_summary: "12 lines", ok: true }),
      ev("loop_completed", { text: "the answer" }),
    ]);

    expect(turn.answer).toBe("the answer");
    expect(turn.thinking).toBe("let me reason");
    expect(turn.thinking).not.toBe(turn.answer);
    expect(turn.toolCalls).toHaveLength(1);
    expect(turn.toolCalls[0]).toMatchObject({
      toolName: "read_file",
      argsSummary: "path=a.py",
      status: "completed",
      ok: true,
      resultSummary: "12 lines",
    });
  });

  it("keeps a thinking_delta after a token_delta out of the answer", () => {
    const turn = foldStreamedTurn([
      ev("token_delta", { text: "ans" }),
      ev("thinking_delta", { text: "more reasoning" }),
      ev("token_delta", { text: "wer" }),
    ]);
    expect(turn.answer).toBe("answer");
    expect(turn.thinking).toBe("more reasoning");
  });

  it("leaves a started tool call without a completion in the started state", () => {
    const turn = foldStreamedTurn([
      ev("tool_call_started", { tool_name: "list_runs", args_summary: "" }),
    ]);
    expect(turn.toolCalls).toHaveLength(1);
    expect(turn.toolCalls[0].status).toBe("started");
    expect(turn.toolCalls[0].ok).toBeNull();
  });

  it("falls back to loop_completed.text when no token_delta streamed", () => {
    const turn = foldStreamedTurn([
      ev("tool_call_started", { tool_name: "noop" }),
      ev("loop_completed", { text: "final summary" }),
    ]);
    expect(turn.answer).toBe("final summary");
  });

  it("returns an empty/neutral turn for an empty event list", () => {
    expect(foldStreamedTurn([])).toEqual({ answer: "", thinking: "", toolCalls: [] });
  });

  it("matches a completed tool call to its started entry by tool_name (FIFO)", () => {
    const turn = foldStreamedTurn([
      ev("tool_call_started", { tool_name: "a" }),
      ev("tool_call_started", { tool_name: "b" }),
      ev("tool_call_completed", { tool_name: "b", ok: false, result_summary: "boom" }),
    ]);
    const a = turn.toolCalls.find((t) => t.toolName === "a");
    const b = turn.toolCalls.find((t) => t.toolName === "b");
    expect(a?.status).toBe("started");
    expect(b?.status).toBe("completed");
    expect(b?.ok).toBe(false);
    expect(b?.resultSummary).toBe("boom");
  });
});
