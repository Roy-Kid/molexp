/**
 * Pure presentation core of the workspace activity feed (vision-loop-12).
 *
 * The component (`WorkspaceActivityFeed.tsx`) stays a thin fetch-and-render
 * shell; everything decidable — event-type → icon/label mapping and ref →
 * entity-link resolution — lives here so it is unit-testable in the node env.
 */

import {
  BookOpen,
  CheckCircle2,
  CircleDot,
  FileBox,
  FlaskConical,
  PlusCircle,
  Workflow,
  XCircle,
} from "lucide-react";

export interface WorkspaceEventRow {
  id: string;
  seq: number;
  type: string;
  actor: string;
  created_at: string;
  payload: Record<string, unknown>;
  refs: string[];
}

export interface EventVisual {
  icon: typeof CircleDot;
  label: string;
  dotClass: string;
}

const VISUALS: Record<string, EventVisual> = {
  "run.created": { icon: PlusCircle, label: "Run created", dotClass: "bg-muted-foreground/40" },
  "run.started": { icon: CircleDot, label: "Run started", dotClass: "bg-info" },
  "run.completed": { icon: CheckCircle2, label: "Run completed", dotClass: "bg-success" },
  "run.failed": { icon: XCircle, label: "Run failed", dotClass: "bg-destructive" },
  "asset.added": { icon: FileBox, label: "Asset added", dotClass: "bg-info" },
  "knowledge.created": { icon: BookOpen, label: "Knowledge created", dotClass: "bg-success" },
  "workflow.created": { icon: Workflow, label: "Workflow created", dotClass: "bg-info" },
  "experiment.created": {
    icon: FlaskConical,
    label: "Experiment created",
    dotClass: "bg-muted-foreground/40",
  },
};

const FALLBACK_VISUAL: EventVisual = {
  icon: CircleDot,
  label: "",
  dotClass: "bg-muted-foreground/40",
};

/** Icon/label/color for an event type; unknown types render their raw name. */
export const eventVisualFor = (type: string): EventVisual =>
  VISUALS[type] ?? { ...FALLBACK_VISUAL, label: type };

export type ResolvedRef =
  | { kind: "run"; runId: string; text: string }
  | { kind: "knowledge"; path: string; text: string }
  | { kind: "plain"; text: string };

/**
 * Resolve one event ref into an entity link when the snapshot knows it.
 *
 * A ref matching a known run id links to the run inspector; a ref of a
 * `knowledge.created` event is a bundle-relative identity path and links to
 * the Knowledge section. Everything else (asset ids, hashes, ids the
 * snapshot no longer carries) stays plain text — never a dead link.
 */
export const resolveEventRef = (
  ref: string,
  eventType: string,
  knownRunIds: ReadonlySet<string>,
  payload?: Record<string, unknown>,
): ResolvedRef => {
  if (knownRunIds.has(ref)) {
    return { kind: "run", runId: ref, text: ref };
  }
  if (eventType === "knowledge.created") {
    return { kind: "knowledge", path: ref, text: ref };
  }
  // An asset id is a bare UUID — the payload carries the human name the
  // emit recorded, so show that (the UUID stays in the hover title).
  if (eventType === "asset.added" && typeof payload?.name === "string" && payload.name) {
    return { kind: "plain", text: payload.name };
  }
  return { kind: "plain", text: ref };
};

export const FEED_EMPTY_TEXT = "No events yet.";
