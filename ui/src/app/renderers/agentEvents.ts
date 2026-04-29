import type { ApiSessionEvent } from "@/app/types";

export interface PendingUserRequest {
  requestId: string;
  prompt: string | null;
}

/**
 * Walks an event log backwards to detect whether the agent is currently
 * waiting on a UserMessageRequestEvent that has not yet been answered
 * by a UserMessageEvent.
 */
export const derivePendingUserRequest = (
  events: ApiSessionEvent[],
): PendingUserRequest | null => {
  for (let i = events.length - 1; i >= 0; i--) {
    const ev = events[i];
    if (ev.type === "UserMessageEvent") return null;
    if (ev.type === "UserMessageRequestEvent") {
      const payload = (ev.payload ?? {}) as Record<string, unknown>;
      const rid = payload.request_id;
      if (typeof rid === "string") {
        return {
          requestId: rid,
          prompt: typeof payload.prompt === "string" ? payload.prompt : null,
        };
      }
      return null;
    }
  }
  return null;
};

export interface ConversationTurn {
  /** Stable key for React lists. */
  key: string;
  /** The question that opened this turn. The first turn uses the goal. */
  question: string;
  /** Whether this turn was opened by the original goal vs a follow-up message. */
  source: "goal" | "user";
  /** The agent's final answer for this turn, if any. */
  result: ApiSessionEvent | null;
  /** Intermediate events (plans, tool calls, observations, …) — collapsed by default. */
  steps: ApiSessionEvent[];
  /** True when this turn is still streaming (no terminal result yet). */
  inProgress: boolean;
}

const isResultEvent = (event: ApiSessionEvent): boolean =>
  event.type === "SessionCompletedEvent" || event.type === "ResultArtifactEvent";

const eventKey = (event: ApiSessionEvent, fallback: number): string =>
  `${event.type}-${event.ts}-${fallback}`;

/**
 * Group events into conversational turns. Each turn is opened by either
 * the original goal (the implicit first turn) or a UserMessageEvent, and
 * closed by the final ResultArtifactEvent / SessionCompletedEvent before
 * the next user message.
 *
 * Intermediate events (tool calls, observations, replans, approvals,
 * workflow runs) are surfaced as `steps` so the UI can collapse them.
 */
export const groupEventsIntoTurns = (
  events: ApiSessionEvent[],
  goalDescription: string,
): ConversationTurn[] => {
  const turns: ConversationTurn[] = [];

  let current: ConversationTurn = {
    key: "turn-goal",
    question: goalDescription,
    source: "goal",
    result: null,
    steps: [],
    inProgress: true,
  };

  events.forEach((event, idx) => {
    if (event.type === "UserMessageEvent") {
      const payload = (event.payload ?? {}) as Record<string, unknown>;
      const content = typeof payload.content === "string" ? payload.content : "";
      // Close the previous turn (it may not have a "result" event if the user
      // interrupted; mark it as no longer in-progress regardless).
      current.inProgress = false;
      turns.push(current);
      current = {
        key: eventKey(event, idx),
        question: content,
        source: "user",
        result: null,
        steps: [],
        inProgress: true,
      };
      return;
    }

    if (isResultEvent(event)) {
      // SessionCompletedEvent / ResultArtifactEvent — promote to the turn's
      // headline result. If multiple show up in one turn, the last one wins
      // for the headline but earlier ones remain as steps so the user can
      // still inspect them.
      if (current.result) {
        current.steps.push(current.result);
      }
      current.result = event;
      current.inProgress = false;
      return;
    }

    current.steps.push(event);
  });

  turns.push(current);
  return turns;
};
