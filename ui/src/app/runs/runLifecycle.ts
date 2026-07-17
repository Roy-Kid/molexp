/**
 * Run lifecycle — single source for status → allowed verbs.
 *
 * Mirrors workspace/server law (three orthogonal verbs + cancel):
 *
 *   pending              → start
 *   running              → cancel
 *   failed | cancelled   → resume | rerun | rerun-fresh
 *   succeeded            → (done; harvest is post-hoc knowledge)
 *
 * Harvest is not a lifecycle verb; it is available on any terminal
 * outcome the backend accepts (succeeded / failed / cancelled).
 */

export const RETRYABLE_STATUSES = new Set(["failed", "cancelled"]);

/** Terminal outcomes that may be interpreted into Knowledge. */
export const HARVESTABLE_STATUSES = new Set(["succeeded", "failed", "cancelled"]);

export const TERMINAL_STATUSES = new Set(["succeeded", "failed", "cancelled", "skipped"]);

export type RunLifecyclePhase = "pending" | "running" | "retryable" | "succeeded" | "other";

export function runPhase(status: string): RunLifecyclePhase {
  const s = status.toLowerCase();
  if (s === "pending") return "pending";
  if (s === "running") return "running";
  if (RETRYABLE_STATUSES.has(s)) return "retryable";
  if (s === "succeeded") return "succeeded";
  return "other";
}

export function canStart(status: string): boolean {
  return status.toLowerCase() === "pending";
}

export function canCancel(status: string): boolean {
  return status.toLowerCase() === "running";
}

export function canResume(status: string): boolean {
  return RETRYABLE_STATUSES.has(status.toLowerCase());
}

export function canRerun(status: string): boolean {
  return RETRYABLE_STATUSES.has(status.toLowerCase());
}

export function canHarvest(status: string): boolean {
  return HARVESTABLE_STATUSES.has(status.toLowerCase());
}

export function isTerminalStatus(status: string): boolean {
  return TERMINAL_STATUSES.has(status.toLowerCase());
}

/** After a verb that kicks off work, land on Executions (live graph). */
export const POST_DISPATCH_TAB = "executions" as const;
