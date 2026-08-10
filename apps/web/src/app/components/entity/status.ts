/** The fixed MolCrafts status vocabulary, plus wire-alias normalization. */

export type CanonicalStatus =
  | "draft"
  | "ready"
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled"
  | "cached"
  | "warning";

const STATUS_ALIASES: Record<string, CanonicalStatus> = {
  draft: "draft",
  idle: "ready",
  ready: "ready",
  active: "ready",
  created: "queued",
  pending: "queued",
  queued: "queued",
  scheduled: "queued",
  submitted: "queued",
  in_progress: "running",
  retrying: "running",
  running: "running",
  completed: "completed",
  done: "completed",
  granted: "completed",
  ok: "completed",
  passed: "completed",
  succeeded: "completed",
  success: "completed",
  approved: "completed",
  failed: "failed",
  error: "failed",
  invalid: "failed",
  lost: "failed",
  rejected: "failed",
  timed_out: "failed",
  aborted: "cancelled",
  cancelled: "cancelled",
  canceled: "cancelled",
  archived: "cancelled",
  expired: "cancelled",
  killed: "cancelled",
  skipped: "cancelled",
  stopped: "cancelled",
  superseded: "cancelled",
  user_cancelled: "cancelled",
  cached: "cached",
  warning: "warning",
  awaiting_approval: "warning",
  awaiting_review: "warning",
  awaiting_user: "warning",
  blocked: "warning",
  conflicting: "warning",
  paused: "warning",
  stale: "warning",
  waiting_approval: "warning",
  waiting_for_approval: "warning",
  waiting_for_review: "warning",
};

export const canonicalStatusFor = (status: string | null | undefined): CanonicalStatus | null => {
  if (!status) return null;
  return STATUS_ALIASES[status.toLowerCase()] ?? "ready";
};
