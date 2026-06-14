/**
 * Shared time formatters used across the app for run/execution rows,
 * scheduler dashboards, and other timeline views.
 */

export const formatDuration = (seconds: number | null): string => {
  if (seconds === null || Number.isNaN(seconds) || seconds < 0) {
    return "—";
  }
  const total = Math.floor(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) {
    return `${h}h ${m.toString().padStart(2, "0")}m ${s.toString().padStart(2, "0")}s`;
  }
  if (m > 0) {
    return `${m}m ${s.toString().padStart(2, "0")}s`;
  }
  return `${s}s`;
};

/**
 * Compact duration for inline rows (tool calls, turn footers): sub-10s keeps
 * one decimal ("0.8s"), sub-minute rounds to whole seconds ("42s"), longer
 * spans collapse to "1m07s" / "2h05m". Returns "" for unusable input so
 * callers can simply skip rendering.
 */
export const formatDurationCompact = (seconds: number | null): string => {
  if (seconds === null || Number.isNaN(seconds) || seconds < 0) {
    return "";
  }
  if (seconds < 10) {
    return `${seconds.toFixed(1)}s`;
  }
  if (seconds < 60) {
    return `${Math.round(seconds)}s`;
  }
  const total = Math.floor(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) {
    return `${h}h${m.toString().padStart(2, "0")}m`;
  }
  return `${m}m${s.toString().padStart(2, "0")}s`;
};

export const formatRelative = (iso: string | null): string => {
  if (!iso) return "—";
  const ts = new Date(iso).getTime();
  if (Number.isNaN(ts)) return iso;
  const delta = (Date.now() - ts) / 1000;
  if (delta < 60) return "just now";
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
  return `${Math.floor(delta / 86400)}d ago`;
};

export const formatTimestamp = (iso: string | null): string => {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString();
};
