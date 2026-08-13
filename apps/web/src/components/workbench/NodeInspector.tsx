/**
 * Right-rail inspector chrome for a selected workflow node or entity.
 * Sections + separators — never nested cards.
 */

import { type JSX, type ReactNode, useId } from "react";

import { cn } from "@/lib/utils";

import { RunStatusBadge } from "./RunStatusBadge";

export interface NodeInspectorProps {
  title: string;
  subtitle?: string | null;
  status?: string | null;
  /** Optional mono id (task id, run id, …). */
  identity?: string | null;
  children?: ReactNode;
  footer?: ReactNode;
  className?: string;
  empty?: boolean;
  emptyHint?: string;
}

export const NodeInspector = ({
  title,
  subtitle,
  status,
  identity,
  children,
  footer,
  className,
  empty,
  emptyHint = "Select a node or entity to inspect.",
}: NodeInspectorProps): JSX.Element => {
  const titleId = useId();

  if (empty) {
    return (
      <aside
        aria-label={title}
        className={cn(
          "flex h-full items-center justify-center px-4 text-center text-label text-muted-foreground",
          className,
        )}
      >
        {emptyHint}
      </aside>
    );
  }

  return (
    <aside
      aria-labelledby={titleId}
      className={cn("flex h-full min-h-0 flex-col bg-surface-subtle", className)}
    >
      <header className="flex flex-none flex-col gap-1 border-b border-border px-3 py-2">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <h2 id={titleId} className="truncate text-title font-semibold text-foreground">
              {title}
            </h2>
            {subtitle && <p className="truncate text-label text-muted-foreground">{subtitle}</p>}
          </div>
          {status != null && status !== "" && <RunStatusBadge status={status} />}
        </div>
        {identity && (
          <p className="truncate font-mono text-micro text-muted-foreground tabular-nums">
            {identity}
          </p>
        )}
      </header>
      <div className="min-h-0 flex-1 overflow-auto px-3 py-2">{children}</div>
      {footer && (
        <footer className="flex flex-none flex-wrap items-center gap-2 border-t border-border px-3 py-2">
          {footer}
        </footer>
      )}
    </aside>
  );
};
