import type { JSX } from "react";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

export type StatusTone = "success" | "error" | "running" | "neutral" | "warning";

const STATUS_TONE: Record<string, StatusTone> = {
  active: "success",
  succeeded: "success",
  completed: "success",
  failed: "error",
  error: "error",
  running: "running",
  pending: "neutral",
  waiting_for_review: "warning",
  approved: "success",
  rejected: "error",
  expired: "neutral",
  archived: "neutral",
  cancelled: "neutral",
  draft: "warning",
  skipped: "warning",
};

const TONE_CLASSES: Record<StatusTone, string> = {
  success: "border-success/25 bg-success-soft text-success-foreground hover:bg-success-soft",
  error: "border-destructive/25 bg-destructive/10 text-destructive hover:bg-destructive/10",
  running: "border-info/25 bg-info-soft text-info-foreground hover:bg-info-soft",
  neutral: "border-border bg-muted text-muted-foreground hover:bg-muted",
  warning: "border-warning/25 bg-warning-soft text-warning-foreground hover:bg-warning-soft",
};

const DOT_CLASSES: Record<StatusTone, string> = {
  success: "bg-success",
  error: "bg-destructive",
  running: "bg-info",
  neutral: "bg-muted-foreground/40",
  warning: "bg-warning",
};

const SIZE_CLASSES: Record<StatusBadgeSize, string> = {
  sm: "px-1.5 py-0 text-[10px] font-medium",
  md: "px-2 py-0.5 text-xs font-medium",
};

export type StatusBadgeSize = "sm" | "md";

export interface StatusBadgeProps {
  status: string | null | undefined;
  size?: StatusBadgeSize;
  pulse?: boolean;
  /** Show a leading colored dot (matching tone). */
  dot?: boolean;
  /** Set false to render only the dot without the text label. */
  showLabel?: boolean;
}

const resolveTone = (status: string | null | undefined): StatusTone => {
  if (!status) return "neutral";
  return STATUS_TONE[status.toLowerCase()] ?? "neutral";
};

export const statusToneFor = resolveTone;
export const statusDotClass = (status: string | null | undefined): string =>
  DOT_CLASSES[resolveTone(status)];

export const StatusBadge = ({
  status,
  size = "md",
  pulse,
  dot = false,
  showLabel = true,
}: StatusBadgeProps): JSX.Element | null => {
  if (!status) return null;
  const tone = resolveTone(status);
  const shouldPulse = pulse ?? tone === "running";
  return (
    <Badge
      variant="outline"
      className={cn(
        TONE_CLASSES[tone],
        SIZE_CLASSES[size],
        "inline-flex items-center gap-1.5",
        shouldPulse && "animate-pulse",
      )}
    >
      {dot && (
        <span
          aria-hidden="true"
          className={cn("inline-block h-1.5 w-1.5 rounded-full", DOT_CLASSES[tone])}
        />
      )}
      {showLabel && <span>{status}</span>}
    </Badge>
  );
};
