import { CheckCircle2, Circle, CircleSlash2, Clock3, LoaderCircle, XCircle } from "lucide-react";
import type { ComponentType, JSX, SVGProps } from "react";
import { cn } from "@/lib/utils";

export type StatusIconTone = "success" | "error" | "running" | "warning" | "neutral";

interface StatusIconMeta {
  icon: ComponentType<SVGProps<SVGSVGElement>>;
  tone: StatusIconTone;
  spin?: boolean;
}

const TONE_CLASS: Record<StatusIconTone, string> = {
  success: "text-success",
  error: "text-destructive",
  running: "text-info",
  warning: "text-warning",
  neutral: "text-muted-foreground",
};

export const statusIconMeta = (status: string | null | undefined): StatusIconMeta => {
  switch (status?.toLowerCase()) {
    case "active":
    case "approved":
    case "completed":
    case "succeeded":
    case "success":
      return { icon: CheckCircle2, tone: "success" };
    case "failed":
    case "error":
    case "lost":
    case "rejected":
    case "timed_out":
      return { icon: XCircle, tone: "error" };
    case "running":
      return { icon: LoaderCircle, tone: "running", spin: true };
    case "draft":
    case "expired":
    case "pending":
    case "queued":
    case "submitted":
    case "waiting_for_review":
      return { icon: Circle, tone: status?.toLowerCase() === "draft" ? "warning" : "neutral" };
    case "cancelled":
    case "skipped":
    case "archived":
      return { icon: CircleSlash2, tone: "neutral" };
    default:
      return { icon: Clock3, tone: "neutral" };
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
  switch (status?.toLowerCase()) {
    case "cancelled":
    case "skipped":
    case "archived":
      return "skipped";
  }
  switch (statusIconMeta(status).tone) {
    case "success":
      return "success";
    case "error":
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
  const text = label ?? status ?? "unknown";
  return (
    <Icon
      aria-label={text}
      role="img"
      className={cn(
        "h-3.5 w-3.5 flex-none",
        TONE_CLASS[meta.tone],
        meta.spin && "animate-spin",
        className,
      )}
    />
  );
};
