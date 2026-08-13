/**
 * Status-gated actions for run *lists* (experiment table, left-nav tree).
 *
 * Context menu is deliberately minimal — open/lifecycle verbs live on the
 * run page / primary row control. Right-click is copy-only (id + absolute path).
 */

import { Copy, FolderOpen, type LucideIcon } from "lucide-react";
import { runPhase } from "@/app/runs/runLifecycle";
import type { RunSummary } from "@/app/types";

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
  copyId: (run: RunSummary) => void;
  /** Absolute / host-qualified path (see {@link formatQualifiedPath}). */
  copyPath: (run: RunSummary) => void;
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

/**
 * Context-menu / row-menu for one run — copy id + copy absolute path only.
 */
export function buildRunListActions(run: RunSummary, handlers: RunListHandlers): RunListAction[] {
  return [
    {
      id: "copy-id",
      label: "Copy ID",
      icon: Copy,
      onSelect: () => handlers.copyId(run),
    },
    {
      id: "copy-path",
      label: "Copy path",
      icon: FolderOpen,
      title: "Absolute path (host-qualified when remote)",
      onSelect: () => handlers.copyPath(run),
    },
  ];
}
