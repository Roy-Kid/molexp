// ─────────────────────────────────────────────────────────────────────────────
// Entity model — the single source of truth for "what kinds of things exist and
// how they look". Every navigable thing in the workspace is an ``EntityRef``:
// a (kind, id) pair plus optional carried context. Icons, labels, and accent
// classes live here once, so the left nav, breadcrumb, related panel, and
// command palette all render an entity identically no matter where it surfaces.
// ─────────────────────────────────────────────────────────────────────────────

import {
  Archive,
  Blocks,
  BookOpen,
  Bot,
  Box,
  FileText,
  FlaskConical,
  PlayCircle,
  Workflow as WorkflowIcon,
} from "lucide-react";
import type { ComponentType } from "react";
import type { SemanticObjectType } from "@/app/types";

// EntityKind is intentionally identical to the renderer-layer SemanticObjectType
// so an EntityRef and a Selection are trivially interconvertible during the
// transition. "workspace-file" is the on-disk file kind.
export type EntityKind = SemanticObjectType;

export interface EntityRef {
  kind: EntityKind;
  id: string;
  /** Owning run id — required to address a ``task`` (graph node) uniquely. */
  runId?: string;
  /** Precomputed display label, so a ref can render without a snapshot lookup. */
  label?: string;
  /** Precomputed status, for the badge on related/nav rows. */
  status?: string;
}

export interface EntityKindMeta {
  kind: EntityKind;
  label: string;
  plural: string;
  icon: ComponentType<{ className?: string }>;
  /** Entity type is carried by the Lucide glyph; its colour stays neutral. */
  iconClassName: string;
}

export const ENTITY_META: Record<EntityKind, EntityKindMeta> = {
  project: {
    kind: "project",
    label: "Project",
    plural: "Projects",
    icon: Blocks,
    iconClassName: "text-muted-foreground",
  },
  experiment: {
    kind: "experiment",
    label: "Experiment",
    plural: "Experiments",
    icon: FlaskConical,
    iconClassName: "text-muted-foreground",
  },
  run: {
    kind: "run",
    label: "Run",
    plural: "Runs",
    icon: PlayCircle,
    iconClassName: "text-muted-foreground",
  },
  task: {
    kind: "task",
    label: "Task",
    plural: "Tasks",
    icon: Box,
    iconClassName: "text-muted-foreground",
  },
  workflow: {
    kind: "workflow",
    label: "Workflow",
    plural: "Workflows",
    icon: WorkflowIcon,
    iconClassName: "text-muted-foreground",
  },
  asset: {
    kind: "asset",
    label: "Asset",
    plural: "Assets",
    icon: Archive,
    iconClassName: "text-muted-foreground",
  },
  agent: {
    kind: "agent",
    label: "Agent Task",
    plural: "Agent Tasks",
    icon: Bot,
    iconClassName: "text-muted-foreground",
  },
  "workspace-file": {
    kind: "workspace-file",
    label: "File",
    plural: "Files",
    icon: FileText,
    iconClassName: "text-muted-foreground",
  },
  knowledge: {
    kind: "knowledge",
    label: "Knowledge",
    plural: "Knowledge",
    icon: BookOpen,
    iconClassName: "text-muted-foreground",
  },
};

export const entityMeta = (kind: EntityKind): EntityKindMeta => ENTITY_META[kind];
