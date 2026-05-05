/**
 * Pure helpers used by AgentSessionInspector.
 *
 * Lives in its own file so the unit tests can run in the node test
 * environment without pulling in `agentApi` / `planApi` (which trigger
 * a transitive `state/api` module-load).
 *
 * Per the agent-harness UI lockstep spec (§8 + R4):
 *
 * - Phase 5 ships a one-shot migration tool. Pre-migration sessions on
 *   disk get tombstoned with `status === "legacy"`.
 * - The inspector treats legacy sessions as read-only: badge them in
 *   the header and disable any mutating action. There are no mutating
 *   actions in the inspector today, but we want the helper centralised
 *   so call sites elsewhere (resume / cancel / approve buttons in
 *   AgentViewer) can adopt it without re-deriving the rule.
 */

import type { ApiAgentSession } from "@/app/types";

export const LEGACY_SESSION_STATUS = "legacy" as const;

export const isLegacySession = (session: ApiAgentSession | null | undefined): boolean =>
  session?.status === LEGACY_SESSION_STATUS;

export interface LegacyBadgeMeta {
  /** Visible badge label. */
  readonly label: string;
  /** Tooltip explaining why this session is read-only. */
  readonly tooltip: string;
}

export const legacyBadgeMeta = (): LegacyBadgeMeta => ({
  label: "legacy · read-only",
  tooltip:
    "Pre-migration session — message history unavailable. Read-only since the .molexp-agent/ migration.",
});
