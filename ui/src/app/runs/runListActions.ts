/**
 * Status-gated actions for run *lists* (experiment table, left-nav tree).
 * Same verb law as {@link runLifecycle} / RunToolbar — menus never show a
 * disabled Cancel for non-running rows.
 */

import {
  Ban,
  BookMarked,
  Copy,
  ExternalLink,
  GitBranch,
  type LucideIcon,
  Play,
  RefreshCw,
  Terminal,
} from "lucide-react";
import {
  canCancel,
  canHarvest,
  canRerun,
  canResume,
  canStart,
  runPhase,
} from "@/app/runs/runLifecycle";
import type { ObjectView, RunSummary } from "@/app/types";

export interface RunListAction {
  id: string;
  label: string;
  icon?: LucideIcon;
  disabled?: boolean;
  destructive?: boolean;
  separatorBefore?: boolean;
  title?: string;
  onSelect: () => void;
}

export interface RunListHandlers {
  open: (run: RunSummary, view?: ObjectView) => void;
  cancel: (run: RunSummary) => void;
  /** In-place resume (failed/cancelled only). */
  resume: (run: RunSummary) => void;
  /** New execution; `fresh` bypasses cache. */
  rerun: (run: RunSummary, fresh?: boolean) => void;
  copyId: (run: RunSummary) => void;
  /**
   * When set, lifecycle verbs (start/resume/rerun/cancel/harvest) are disabled
   * and this string is shown as the hover tip (role / auth denial).
   */
  writeDeniedReason?: string | null;
}

/** Compact primary control for a table row (one button max). */
export type RunPrimaryVerb =
  | { kind: "start"; label: "Start" }
  | { kind: "cancel"; label: "Cancel" }
  | { kind: "resume"; label: "Resume" }
  | null;

export function primaryRunVerb(status: string): RunPrimaryVerb {
  switch (runPhase(status)) {
    case "pending":
      return { kind: "start", label: "Start" };
    case "running":
      return { kind: "cancel", label: "Cancel" };
    case "retryable":
      return { kind: "resume", label: "Resume" };
    default:
      return null;
  }
}

function gateWrite(action: RunListAction, denied: string | null | undefined): RunListAction {
  if (!denied) return action;
  return { ...action, disabled: true, title: denied };
}

/**
 * Full context-menu / row-menu for one run.
 * Lifecycle verbs only appear when allowed; utilities always available.
 */
export function buildRunListActions(run: RunSummary, handlers: RunListHandlers): RunListAction[] {
  const denied = handlers.writeDeniedReason ?? null;
  const actions: RunListAction[] = [
    {
      id: "open",
      label: "Open",
      icon: ExternalLink,
      onSelect: () => handlers.open(run),
    },
    {
      id: "executions",
      label: "Executions",
      icon: GitBranch,
      onSelect: () => handlers.open(run, "executions"),
    },
    {
      id: "logs",
      label: "Logs",
      icon: Terminal,
      onSelect: () => handlers.open(run, "logs"),
    },
  ];

  if (canStart(run.status)) {
    actions.push(
      gateWrite(
        {
          id: "start",
          label: "Start",
          icon: Play,
          separatorBefore: true,
          title: "Open run to start",
          onSelect: () => handlers.open(run),
        },
        denied,
      ),
    );
  }

  if (canResume(run.status)) {
    actions.push(
      gateWrite(
        {
          id: "resume",
          label: "Resume",
          icon: Play,
          separatorBefore: true,
          title: "Continue last execution",
          onSelect: () => handlers.resume(run),
        },
        denied,
      ),
    );
  }

  if (canRerun(run.status)) {
    actions.push(
      gateWrite(
        {
          id: "rerun",
          label: "Rerun",
          icon: RefreshCw,
          title: "New execution from top",
          onSelect: () => handlers.rerun(run, false),
        },
        denied,
      ),
    );
    actions.push(
      gateWrite(
        {
          id: "rerun-fresh",
          label: "Fresh",
          icon: RefreshCw,
          title: "Rerun without cache",
          onSelect: () => handlers.rerun(run, true),
        },
        denied,
      ),
    );
  }

  if (canCancel(run.status)) {
    actions.push(
      gateWrite(
        {
          id: "cancel",
          label: "Cancel",
          icon: Ban,
          destructive: true,
          separatorBefore: true,
          onSelect: () => handlers.cancel(run),
        },
        denied,
      ),
    );
  }

  if (canHarvest(run.status)) {
    actions.push(
      gateWrite(
        {
          id: "harvest",
          label: "Harvest",
          icon: BookMarked,
          separatorBefore: true,
          title: "Open run to harvest",
          onSelect: () => handlers.open(run),
        },
        denied,
      ),
    );
  }

  actions.push({
    id: "copy-id",
    label: "Copy ID",
    icon: Copy,
    separatorBefore: true,
    onSelect: () => handlers.copyId(run),
  });

  return actions;
}
