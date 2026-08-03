/**
 * Product chrome for one workflow DAG node.
 *
 * Graph engines own drag, ports, and edges. This component owns the node's
 * domain language: role, task type, execution status, validation, and focus.
 */

import {
  Circle,
  CircleDot,
  GitBranch,
  Layers3,
  type LucideIcon,
  Repeat2,
  Square,
  Workflow,
} from "lucide-react";
import type { JSX, ReactNode } from "react";

import { cn } from "@/lib/utils";

import { RunStatusBadge } from "./RunStatusBadge";

export type WorkflowNodeRole = "action" | "initial" | "final" | "branch" | "loop";
export type WorkflowNodeCanvasRole = "input" | "output" | "task";

export interface WorkflowNodeProps {
  title: string;
  taskType?: string | null;
  status?: string | null;
  role?: WorkflowNodeRole;
  canvasRole?: WorkflowNodeCanvasRole;
  parallel?: boolean;
  subworkflow?: boolean;
  error?: string | null;
  selected?: boolean;
  onActivate?: () => void;
  footer?: ReactNode;
  className?: string;
  density?: "default" | "compact";
}

const ROLE_ICON: Record<WorkflowNodeRole, LucideIcon> = {
  action: Square,
  initial: CircleDot,
  final: Circle,
  branch: GitBranch,
  loop: Repeat2,
};

const CANVAS_ICON: Record<WorkflowNodeCanvasRole, LucideIcon> = {
  input: CircleDot,
  output: Circle,
  task: Square,
};

export const WorkflowNode = ({
  title,
  taskType,
  status,
  role = "action",
  canvasRole,
  parallel,
  subworkflow,
  error,
  selected,
  onActivate,
  footer,
  className,
  density,
}: WorkflowNodeProps): JSX.Element => {
  const compact = density === "compact" || Boolean(canvasRole);
  const NodeRoot = onActivate ? "button" : "div";
  const RoleIcon = canvasRole ? CANVAS_ICON[canvasRole] : ROLE_ICON[role];

  return (
    <NodeRoot
      type={onActivate ? "button" : undefined}
      onClick={onActivate}
      aria-current={selected ? "true" : undefined}
      className={cn(
        "relative w-full rounded-[var(--radius-panel)] border border-border bg-surface text-left shadow-none",
        "transition-[background-color,border-color] duration-[var(--motion-base)] ease-[var(--motion-ease)]",
        compact
          ? "min-w-[9.5rem] max-w-[16.25rem] px-3 py-2"
          : "min-w-[13.75rem] max-w-[17.5rem] p-3",
        selected && "border-accent/60 bg-accent/5 ring-1 ring-accent/20",
        onActivate &&
          "cursor-pointer hover:bg-interactive/40 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
        className,
      )}
    >
      <div className="flex min-w-0 items-center gap-2">
        <RoleIcon
          aria-hidden
          strokeWidth={1.5}
          className="h-4 w-4 flex-none text-muted-foreground"
        />
        <span className="min-w-0 flex-1 truncate text-body font-medium text-foreground">
          {title}
        </span>
        {status && <RunStatusBadge status={status} showLabel={!compact} size="sm" />}
      </div>

      {taskType && (
        <p className="mt-1 truncate pl-6 font-mono text-micro text-muted-foreground">{taskType}</p>
      )}

      {(parallel || subworkflow) && (
        <div className="mt-2 flex flex-wrap items-center gap-2 pl-6 text-micro text-muted-foreground">
          {parallel && (
            <span className="inline-flex items-center gap-1">
              <Layers3 aria-hidden strokeWidth={1.5} className="h-3.5 w-3.5" />
              parallel
            </span>
          )}
          {subworkflow && (
            <span className="inline-flex items-center gap-1">
              <Workflow aria-hidden strokeWidth={1.5} className="h-3.5 w-3.5" />
              subflow
            </span>
          )}
        </div>
      )}

      {error && (
        <p className="mt-2 line-clamp-2 border-t border-status-failed/30 pt-2 font-mono text-micro leading-tight text-status-failed-foreground">
          {error}
        </p>
      )}

      {footer && <div className="mt-2 border-t border-border/70 pt-2">{footer}</div>}
    </NodeRoot>
  );
};
