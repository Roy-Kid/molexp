import type { JSX } from "react";

export type StatusTone = "success" | "error" | "running" | "neutral" | "warning";

const STATUS_TONE: Record<string, StatusTone> = {
  active: "success",
  succeeded: "success",
  completed: "success",
  failed: "error",
  error: "error",
  running: "running",
  pending: "neutral",
  archived: "neutral",
  cancelled: "neutral",
  draft: "warning",
  skipped: "warning",
};

const TONE_SOFT: Record<StatusTone, string> = {
  success: "border-success/25 bg-success-soft text-success-foreground",
  error: "border-destructive/25 bg-destructive/10 text-destructive",
  running: "border-info/25 bg-info-soft text-info-foreground",
  neutral: "border-border bg-muted text-muted-foreground",
  warning: "border-warning/25 bg-warning-soft text-warning-foreground",
};

const SIZE_CLASS: Record<StatusBadgeSize, string> = {
  sm: "px-1.5 py-0 text-[10px]",
  md: "px-2 py-0.5 text-xs",
};

export type StatusBadgeSize = "sm" | "md";

export interface StatusBadgeProps {
  status: string;
  size?: StatusBadgeSize;
  pulse?: boolean;
}

const resolveTone = (status: string): StatusTone => {
  return STATUS_TONE[status.toLowerCase()] ?? "neutral";
};

export const StatusBadge = ({ status, size = "md", pulse }: StatusBadgeProps): JSX.Element => {
  const tone = resolveTone(status);
  const shouldPulse = pulse ?? tone === "running";
  return (
    <span
      className={`inline-flex items-center rounded-full border font-medium ${TONE_SOFT[tone]} ${SIZE_CLASS[size]} ${shouldPulse ? "animate-pulse" : ""}`}
    >
      {status}
    </span>
  );
};
