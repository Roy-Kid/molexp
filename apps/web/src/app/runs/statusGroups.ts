/**
 * Canonical mapping from raw run/execution status strings to UI groups.
 * Both the aggregate functions and the visual status bar consume these
 * — keeping the mapping in one place avoids drift.
 */

export type StatusGroupId = "running" | "pending" | "succeeded" | "failed" | "cancelled";

export interface StatusGroupSpec {
  id: StatusGroupId;
  label: string;
  color: string;
  /** Status strings (lowercase) that belong to this group. */
  aliases: readonly string[];
  /** Canonical value used when this group is applied as a filter. */
  filterValue: string;
}

/** CSS color values from the constitution status ramp (oklch via custom props). */
export const STATUS_GROUPS: readonly StatusGroupSpec[] = [
  {
    id: "running",
    label: "Running",
    color: "var(--status-running)",
    aliases: ["running"],
    filterValue: "running",
  },
  {
    id: "pending",
    label: "Queued",
    color: "var(--status-queued)",
    aliases: ["pending", "queued", "submitted"],
    filterValue: "pending",
  },
  {
    id: "succeeded",
    label: "Completed",
    color: "var(--status-completed)",
    aliases: ["succeeded", "completed", "success"],
    filterValue: "succeeded",
  },
  {
    id: "failed",
    label: "Failed",
    color: "var(--status-failed)",
    aliases: ["failed", "timed_out", "lost", "error"],
    filterValue: "failed",
  },
  {
    id: "cancelled",
    label: "Cancelled",
    color: "var(--status-cancelled)",
    aliases: ["cancelled", "skipped"],
    filterValue: "cancelled",
  },
];

const ALIAS_TO_GROUP: ReadonlyMap<string, StatusGroupId> = (() => {
  const map = new Map<string, StatusGroupId>();
  for (const group of STATUS_GROUPS) {
    for (const alias of group.aliases) map.set(alias, group.id);
  }
  return map;
})();

export const groupForStatus = (status: string | null | undefined): StatusGroupId | null => {
  if (!status) return null;
  return ALIAS_TO_GROUP.get(status.toLowerCase()) ?? null;
};

export const isStatusInGroup = (status: string, groupId: StatusGroupId): boolean =>
  groupForStatus(status) === groupId;
