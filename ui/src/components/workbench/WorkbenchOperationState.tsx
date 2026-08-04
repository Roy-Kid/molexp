/**
 * Single visual + live-region surface for workbench operation feedback.
 * loading · empty · error · running · success · disabled
 */

import { AlertCircle, CheckCircle2 } from "lucide-react";
import type { ReactNode } from "react";
import { ProgressSpinner } from "@/components/ui/progress-spinner";

import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

export type WorkbenchOperationKind =
  | "idle"
  | "loading"
  | "empty"
  | "error"
  | "running"
  | "success"
  | "disabled";

export interface WorkbenchOperationStateProps {
  kind: WorkbenchOperationKind;
  /** Short title (empty / error / success). */
  title?: string;
  /** Supporting detail. */
  detail?: string;
  /** Retry or primary recovery control. */
  action?: ReactNode;
  /** When kind=loading, optional skeleton rows instead of spinner. */
  skeletonRows?: number;
  /** Content when kind is idle / running with body, or success that shows result. */
  children?: ReactNode;
  className?: string;
  /** Compact density for panels and bottom slots; inline for toolbar-sized feedback. */
  density?: "default" | "compact" | "inline";
}

const SKELETON_ROW_KEYS = [
  "first",
  "second",
  "third",
  "fourth",
  "fifth",
  "sixth",
  "seventh",
  "eighth",
] as const;

export const WorkbenchOperationState = ({
  kind,
  title,
  detail,
  action,
  skeletonRows = 0,
  children,
  className,
  density = "default",
}: WorkbenchOperationStateProps): ReactNode => {
  if (kind === "idle") {
    return children ?? null;
  }

  if (kind === "loading") {
    if (skeletonRows > 0) {
      return (
        <div
          className={cn("space-y-2 p-3", className)}
          role="status"
          aria-live="polite"
          aria-atomic="true"
          aria-busy="true"
          aria-label={title ?? "Loading"}
          data-operation-state={kind}
        >
          {SKELETON_ROW_KEYS.slice(0, skeletonRows).map((rowKey, index) => (
            <Skeleton
              key={rowKey}
              className={cn("h-control-compact w-full rounded-control", index === 0 && "w-2/3")}
            />
          ))}
        </div>
      );
    }
    return (
      <span
        className={cn(
          "inline-flex items-center gap-2 text-muted-foreground",
          density === "inline"
            ? "text-micro"
            : density === "compact"
              ? "px-3 py-2 text-label"
              : "px-4 py-6 text-body",
          className,
        )}
        role="status"
        aria-live="polite"
        aria-atomic="true"
        aria-busy="true"
        data-operation-state={kind}
      >
        <ProgressSpinner className="text-status-running" label={title ?? "Loading"} />
        <span>{title ?? "Loading…"}</span>
      </span>
    );
  }

  if (kind === "running") {
    if (density === "inline") {
      return (
        <span
          className={cn(
            "inline-flex items-center gap-2 text-micro text-status-running-foreground",
            className,
          )}
          role="status"
          aria-live="polite"
          aria-atomic="true"
          aria-busy="true"
          data-operation-state={kind}
        >
          <ProgressSpinner className="text-status-running" label={title ?? "Running"} />
          <span>{title ?? "Running…"}</span>
          {detail && <span className="truncate text-muted-foreground">· {detail}</span>}
          {children}
        </span>
      );
    }
    return (
      <div
        className={cn("flex min-h-0 flex-col", className)}
        aria-busy="true"
        data-operation-state={kind}
      >
        <div
          className="flex flex-none items-center gap-2 border-b border-status-running/25 bg-status-running-soft px-3 py-1 text-label text-status-running-foreground"
          role="status"
          aria-live="polite"
          aria-atomic="true"
        >
          <ProgressSpinner className="text-status-running" label={title ?? "Running"} />
          <span>{title ?? "Running…"}</span>
          {detail && <span className="truncate text-muted-foreground">· {detail}</span>}
        </div>
        {children && <div className="min-h-0 flex-1">{children}</div>}
      </div>
    );
  }

  if (kind === "empty") {
    if (density === "inline") {
      return (
        <span
          className={cn(
            "inline-flex items-center gap-2 text-micro text-muted-foreground",
            className,
          )}
          role="status"
          aria-live="polite"
          aria-atomic="true"
          data-operation-state={kind}
        >
          <span>{title ?? "Nothing here"}</span>
          {detail && <span>· {detail}</span>}
          {action}
        </span>
      );
    }
    return (
      <div
        className={cn(
          "flex flex-col items-start gap-1 text-muted-foreground",
          density === "compact" ? "px-3 py-2" : "px-4 py-8",
          className,
        )}
        role="status"
        aria-live="polite"
        aria-atomic="true"
        data-operation-state={kind}
      >
        <p className="text-body font-medium text-foreground/80">{title ?? "Nothing here"}</p>
        {detail && <p className="text-label">{detail}</p>}
        {action && <div className="mt-2">{action}</div>}
      </div>
    );
  }

  if (kind === "error") {
    if (density === "inline") {
      return (
        <span
          className={cn(
            "inline-flex items-center gap-2 text-micro text-status-failed-foreground",
            className,
          )}
          role="alert"
          aria-atomic="true"
          data-operation-state={kind}
        >
          <AlertCircle className="size-3.5 shrink-0" aria-hidden />
          <span>{title ?? "Something failed"}</span>
          {detail && <span className="truncate text-muted-foreground">· {detail}</span>}
          {action}
        </span>
      );
    }
    return (
      <div
        className={cn(
          "flex flex-col items-start gap-2 rounded-panel border border-status-failed/30 bg-status-failed-soft",
          density === "compact" ? "m-2 px-3 py-2" : "m-3 px-3 py-3",
          className,
        )}
        role="alert"
        aria-atomic="true"
        data-operation-state={kind}
      >
        <div className="flex items-start gap-2">
          <AlertCircle
            className="mt-1 size-3.5 shrink-0 text-status-failed-foreground"
            aria-hidden
          />
          <div className="min-w-0 space-y-1">
            <p className="text-body font-medium text-status-failed-foreground">
              {title ?? "Something failed"}
            </p>
            {detail && (
              <p className="font-mono text-label text-status-failed-foreground/90 break-words">
                {detail}
              </p>
            )}
          </div>
        </div>
        {action && <div className="pl-6">{action}</div>}
      </div>
    );
  }

  if (kind === "success") {
    return (
      <span
        className={cn(
          "inline-flex items-center gap-2 text-status-completed-foreground",
          density === "inline"
            ? "text-micro"
            : density === "compact"
              ? "px-3 py-2 text-label"
              : "px-3 py-2 text-body",
          className,
        )}
        role="status"
        aria-live="polite"
        aria-atomic="true"
        data-operation-state={kind}
      >
        <CheckCircle2 className="size-3.5 shrink-0" aria-hidden />
        <span>{title ?? "Done"}</span>
        {detail && <span className="text-muted-foreground">· {detail}</span>}
        {children}
      </span>
    );
  }

  // disabled
  return (
    <span
      className={cn(
        density === "inline" ? "text-micro" : "px-3 py-2 text-label",
        "text-muted-foreground opacity-70",
        className,
      )}
      aria-disabled="true"
      data-operation-state={kind}
    >
      {title ?? "Unavailable"}
      {detail && <span className="block text-micro">{detail}</span>}
    </span>
  );
};
