// ─────────────────────────────────────────────────────────────────────────────
// Entity model — the single source of truth for "what kinds of things exist and
// how they look". Every navigable thing in the workspace is an ``EntityRef``:
// a (kind, id) pair plus optional carried context. Icons, labels, and accent
// colors live here once, so the left nav, breadcrumb, related panel, and
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
  /** Tailwind text-color class for the icon, keyed per kind for at-a-glance ID. */
  accent: string;
}

export const ENTITY_META: Record<EntityKind, EntityKindMeta> = {
  project: {
    kind: "project",
    label: "Project",
    plural: "Projects",
    icon: Blocks,
    accent: "text-blue-500",
  },
  experiment: {
    kind: "experiment",
    label: "Experiment",
    plural: "Experiments",
    icon: FlaskConical,
    accent: "text-purple-500",
  },
  run: {
    kind: "run",
    label: "Run",
    plural: "Runs",
    icon: PlayCircle,
    accent: "text-emerald-500",
  },
  task: {
    kind: "task",
    label: "Task",
    plural: "Tasks",
    icon: Box,
    accent: "text-teal-500",
  },
  workflow: {
    kind: "workflow",
    label: "Workflow",
    plural: "Workflows",
    icon: WorkflowIcon,
    accent: "text-sky-500",
  },
  asset: {
    kind: "asset",
    label: "Asset",
    plural: "Assets",
    icon: Archive,
    accent: "text-amber-500",
  },
  agent: {
    kind: "agent",
    label: "Agent Task",
    plural: "Agent Tasks",
    icon: Bot,
    accent: "text-violet-500",
  },
  "workspace-file": {
    kind: "workspace-file",
    label: "File",
    plural: "Files",
    icon: FileText,
    accent: "text-muted-foreground",
  },
  knowledge: {
    kind: "knowledge",
    label: "Knowledge",
    plural: "Knowledge",
    icon: BookOpen,
    accent: "text-amber-500",
  },
};

export const entityMeta = (kind: EntityKind): EntityKindMeta => ENTITY_META[kind];
