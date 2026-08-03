/**
 * Compact workbench annotation for metadata, categories, and outcomes.
 *
 * Feature code names the meaning; the base Badge treatment stays private.
 */

import type { HTMLAttributes, JSX } from "react";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

export type WorkbenchTagMeaning =
  | "category"
  | "metadata"
  | "selection"
  | "completed"
  | "failed"
  | "warning";

const MEANING_CLASS: Record<WorkbenchTagMeaning, string> = {
  category: "bg-muted text-muted-foreground",
  metadata: "bg-transparent text-foreground",
  selection: "bg-accent/10 text-accent",
  completed: "bg-status-completed-soft text-status-completed-foreground",
  failed: "bg-status-failed-soft text-status-failed-foreground",
  warning: "bg-status-warning-soft text-status-warning-foreground",
};

export interface WorkbenchTagProps extends HTMLAttributes<HTMLDivElement> {
  meaning?: WorkbenchTagMeaning;
  density?: "compact" | "default";
  mono?: boolean;
}

export const WorkbenchTag = ({
  meaning = "category",
  density = "compact",
  mono,
  className,
  ...props
}: WorkbenchTagProps): JSX.Element => (
  <Badge
    variant="outline"
    className={cn(
      "rounded-[var(--radius-control)] font-medium",
      density === "compact" ? "px-2 py-0 text-micro" : "px-2 py-1 text-label",
      MEANING_CLASS[meaning],
      mono && "font-mono tabular-nums",
      className,
    )}
    {...props}
  />
);
