import { CheckCircle2, Circle, CircleSlash2, Clock3, LoaderCircle, XCircle } from "lucide-react";
import type { ComponentType, JSX, SVGProps } from "react";
import { cn } from "@/lib/utils";
import { type CanonicalStatus, canonicalStatusFor } from "./status";

export type StatusIconTone = CanonicalStatus;

interface StatusIconMeta {
  icon: ComponentType<SVGProps<SVGSVGElement>>;
  tone: StatusIconTone;
  label: CanonicalStatus;
  spin?: boolean;
}

const TONE_CLASS: Record<StatusIconTone, string> = {
  draft: "text-status-draft",
  ready: "text-status-ready",
  queued: "text-status-queued",
  running: "text-status-running",
  completed: "text-status-completed",
  failed: "text-status-failed",
  cancelled: "text-status-cancelled",
  cached: "text-status-cached",
  warning: "text-status-warning",
};

export const statusIconMeta = (status: string | null | undefined): StatusIconMeta => {
  const canonical = canonicalStatusFor(status) ?? "ready";
  switch (canonical) {
    case "completed":
      return { icon: CheckCircle2, tone: canonical, label: canonical };
    case "failed":
      return { icon: XCircle, tone: canonical, label: canonical };
    case "running":
      return { icon: LoaderCircle, tone: canonical, label: canonical, spin: true };
    case "draft":
    case "ready":
    case "queued":
      return { icon: Circle, tone: canonical, label: canonical };
    case "cancelled":
      return { icon: CircleSlash2, tone: canonical, label: canonical };
    case "cached":
      return { icon: Clock3, tone: canonical, label: canonical };
    case "warning":
      return { icon: Clock3, tone: canonical, label: canonical };
    default:
      return { icon: Clock3, tone: "ready", label: "ready" };
  }
};

/**
 * The five visual buckets the workflow canvas paints. Shape encodes a node's
 * graph role; this key encodes its execution STATUS as colour:
 *   running → blue · success → green · failed → red · skipped → dashed-grey ·
 *   pending → grey.
 */
export type StatusKey = "running" | "success" | "failed" | "skipped" | "pending";

/**
 * Collapse any backend status string to its {@link StatusKey} colour bucket,
 * reusing {@link statusIconMeta}'s tone mapping so the icon and the body/edge
 * colour never disagree.
 */
export const statusKey = (status: string | null | undefined): StatusKey => {
  switch (canonicalStatusFor(status)) {
    case "cancelled":
      return "skipped";
    case "completed":
      return "success";
    case "failed":
      return "failed";
    case "running":
      return "running";
    default:
      return "pending"; // neutral + warning (draft/queued/…) read as pending
  }
};

interface StatusIconProps {
  status: string | null | undefined;
  className?: string;
  label?: string;
}

export const StatusIcon = ({ status, className, label }: StatusIconProps): JSX.Element => {
  const meta = statusIconMeta(status);
  const Icon = meta.icon;
  const text = label ?? meta.label;
  return (
    <Icon
      aria-label={text}
      role="img"
      className={cn(
        "h-3.5 w-3.5 flex-none",
        TONE_CLASS[meta.tone],
        meta.spin && "mol-motion-progress-spin",
        className,
      )}
    />
  );
};
