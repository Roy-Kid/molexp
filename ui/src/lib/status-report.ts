/**
 * Lightweight status bus for UI tips that land in the bottom status bar.
 * One notification channel for the workbench — no floating toast cards.
 *
 * Call sites may use {@link reportStatus} directly or the `toast` façade
 * (`@/components/ui/toast`), which routes here.
 */

export type StatusReportType = "info" | "error" | "success" | "warning";

export interface StatusReport {
  text: string;
  type: StatusReportType;
  /** Optional 0–100 progress for long-running work. */
  progress?: number;
}

type StatusListener = (report: StatusReport) => void;

const listeners = new Set<StatusListener>();

/** Publish a one-line tip for the bottom status bar activity region. */
export function reportStatus(
  text: string,
  type: StatusReportType = "info",
  progress?: number,
): void {
  const report: StatusReport = { text, type };
  if (progress !== undefined && Number.isFinite(progress)) {
    report.progress = Math.max(0, Math.min(100, progress));
  }
  for (const listener of listeners) {
    listener(report);
  }
}

/** Subscribe to status reports. Returns an unsubscribe function. */
export function subscribeStatus(listener: StatusListener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** Format optional progress as a compact suffix, e.g. ` 42%`. */
export function formatProgressSuffix(progress?: number): string {
  if (progress === undefined || !Number.isFinite(progress)) return "";
  // Status strip already draws a bar + fraction; keep the line free of a
  // second percentage so "Syncing remote tree (12/153)…" stays readable.
  return "";
}
