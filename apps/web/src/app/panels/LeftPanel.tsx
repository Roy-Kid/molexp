import {
  Activity,
  Archive,
  Ban,
  Blocks,
  BookOpen,
  Bot,
  CloudOff,
  Copy,
  ExternalLink,
  FilePlus,
  FileText,
  FlaskConical,
  Folder,
  FolderOpen,
  FolderPlus,
  FolderTree,
  HardDrive,
  PlayCircle,
  Plus,
  RefreshCw,
  Server,
  Settings,
  Sparkles,
  Trash2,
  Workflow,
} from "lucide-react";
import type { ComponentType, ReactNode, SVGProps } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { ApiError } from "@/api/generated";
import { usePermissions } from "@/app/auth";
import { CreateExperimentDialog } from "@/app/components/CreateExperimentDialog";
import { CreateProjectDialog } from "@/app/components/CreateProjectDialog";
import { CreateRunDialog } from "@/app/components/CreateRunDialog";
import { EMPTY_COPY, StatusBadge } from "@/app/components/entity";
import { DocTree } from "@/app/knowledge/DocTree";
import type { TreeNode, TreeNodeAction } from "@/app/panels/TreeView";
import { TreeView } from "@/app/panels/TreeView";
import { computeFacetCounts } from "@/app/runs/aggregates";
import { parseFilterParams, writeFilterParams } from "@/app/runs/filterParams";
import { RunsFacetPanel } from "@/app/runs/RunsFacetPanel";
import { buildRunListActions } from "@/app/runs/runListActions";
import type { WorkspaceRunsFilters } from "@/app/runs/types";
import { useWorkspaceRuns } from "@/app/runs/useWorkspaceRuns";
import { agentApi, workspaceApi } from "@/app/state/api";
import type {
  AgentSessionSummary,
  AssetSummary,
  ExperimentSummary,
  FileKind,
  LeftPanelView,
  ObjectView,
  ProjectSummary,
  RunSummary,
  Selection,
  SemanticStatus,
  ServedWorkspaceSummary,
  WorkspaceSnapshot,
  WorkspaceTreeNode,
} from "@/app/types";
import { useAlert, useConfirm } from "@/components/ConfirmDialog";
import { usePrompt } from "@/components/PromptDialog";
import { Code as InlineCode } from "@/components/ui/code";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { WorkbenchIconAction, WorkbenchToggleAction } from "@/components/workbench";
import { agentTaskDisplayTitle } from "@/lib/agent-task-title";
import { countLabel } from "@/lib/count-label";
import { getWorkspaceFs } from "@/lib/workspace-fs";
import {
  formatQualifiedPath,
  join as joinWorkspacePath,
  type PathDisplayContext,
  runWorkspaceRelativePath,
} from "@/lib/workspace-path";

const errorDetail = (error: unknown): string => {
  if (error instanceof ApiError) {
    const detail = (error.body as { detail?: unknown } | null)?.detail;
    if (typeof detail === "string" && detail) return detail;
    return error.message;
  }
  return error instanceof Error ? error.message : String(error);
};

interface LeftPanelProps {
  view: LeftPanelView;
  selection: Selection | null;
  snapshot: WorkspaceSnapshot;
  searchQuery?: string;
  onViewChange: (view: LeftPanelView) => void;
  onSelect: (selection: Selection) => void;
  onOpenWorkspace: (path: string, options?: { createIfMissing?: boolean }) => Promise<void>;
  onCreateDirectory: (path: string) => void;
  onCreateFile: (path: string) => void;
  onRefresh: () => void;
  /** Lazy-expand a workspace directory via WorkspaceFs (path = node id). */
  onExpandDirectory?: (dirPath: string) => void;
  /** Lazy-load project → experiments (nav expand). */
  onExpandProject?: (projectId: string) => void;
  /** Lazy-load experiment → runs (nav expand). */
  onExpandExperiment?: (projectId: string, experimentId: string) => void;
  isProjectExpanded?: (projectId: string) => boolean;
  isExperimentExpanded?: (projectId: string, experimentId: string) => boolean;
  /** Bumps after refresh clears lazy entity caches — TreeView re-hydrates open folders. */
  dataEpoch?: number;
}

interface ViewOption {
  id: LeftPanelView;
  label: string;
  icon: ComponentType<SVGProps<SVGSVGElement>>;
}

// Order matters: the primary research flow is Projects → Runs → Workflows →
// Workspaces, with the secondary inventories (Assets, Agent Tasks) trailing.
// Labels match each route's section name (see entities/breadcrumbTrail.ts) and
// surface on the icon rail as tooltip + title + aria-label.
// The projects view's top level is Project (children: Experiment → Run) — never
// label this rail "Experiments", which names the middle tier.
const viewOptions: ViewOption[] = [
  { id: "projects", label: "Projects", icon: Blocks },
  { id: "runs", label: "Runs", icon: PlayCircle },
  { id: "activity", label: "Activity", icon: Activity },
  { id: "workflow", label: "Workflows", icon: Workflow },
  { id: "workspace", label: "Workspace", icon: FolderTree },
  { id: "asset", label: "Assets", icon: Archive },
  { id: "agent", label: "Agent Tasks", icon: Bot },
  { id: "knowledge", label: "Knowledge", icon: BookOpen },
];

const listHeaderByView: Record<LeftPanelView, string> = {
  projects: "Projects",
  workspace: "Workspace",
  runs: "Runs",
  activity: "Activity",
  asset: "Assets",
  workflow: "Workflows",
  agent: "Agents",
  knowledge: "Knowledge",
  settings: "Settings",
};

const fileKindByExtension: Record<string, FileKind> = {
  ".yml": "yaml",
  ".yaml": "yaml",
  ".json": "json",
  ".py": "python",
  ".md": "markdown",
  ".txt": "text",
  ".png": "image",
  ".jpg": "image",
  ".jpeg": "image",
};

const detectFileKind = (path: string | undefined): FileKind => {
  if (!path) return "unknown";
  const parts = path.split(".");
  const last = parts[parts.length - 1];
  const extension = parts.length > 1 && last ? `.${last.toLowerCase()}` : "";
  return fileKindByExtension[extension] ?? "unknown";
};

const statusTextClass = (status: SemanticStatus): string => {
  switch (status) {
    case "active":
    case "approved":
    case "succeeded":
      return "font-medium text-success";
    case "failed":
    case "rejected":
      return "font-medium text-destructive";
    case "running":
      return "font-medium text-info";
    case "draft":
    case "expired":
    case "waiting_for_review":
      return "font-medium text-warning";
    case "archived":
    case "cancelled":
    case "skipped":
      return "text-muted-foreground";
    case "pending":
      return "text-muted-foreground";
  }
};

const copyText = async (text: string): Promise<void> => {
  try {
    await navigator.clipboard.writeText(text);
  } catch (error) {
    console.warn("Failed to copy to clipboard:", error);
  }
};

interface ProjectTreeActions {
  onSelect: (selection: Selection) => void;
  onCreateExperiment: (projectId: string) => void;
  onCreateRun: (experimentId: string) => void;
  onDeleteProject: (projectId: string) => void;
  onDeleteExperiment: (experiment: ExperimentSummary) => void;
  onOpenRunView: (run: RunSummary, view?: ObjectView) => void;
  onCopyText: (text: string) => void;
  /** Qualify absolute / host-remote paths for Copy path. */
  pathContext: PathDisplayContext;
  onRefresh: () => void;
  /** Lazy-load experiments when a project row expands. */
  onExpandProject?: (projectId: string) => void;
  /** Lazy-load runs when an experiment row expands. */
  onExpandExperiment?: (projectId: string, experimentId: string) => void;
  isProjectExpanded?: (projectId: string) => boolean;
  isExperimentExpanded?: (projectId: string, experimentId: string) => boolean;
  /** Role denial tip for mutating tree actions (null = allowed). */
  writeDeniedReason?: string | null;
}

const gateTreeWrite = (
  action: TreeNodeAction,
  writeDeniedReason: string | null | undefined,
): TreeNodeAction => {
  if (!writeDeniedReason) return action;
  return { ...action, disabled: true, title: writeDeniedReason };
};

const buildRunActions = (run: RunSummary, actions: ProjectTreeActions): TreeNodeAction[] =>
  buildRunListActions(run, {
    copyId: (r) => actions.onCopyText(r.id),
    copyPath: (r) =>
      actions.onCopyText(formatQualifiedPath(runWorkspaceRelativePath(r), actions.pathContext)),
  }).map((action) => ({
    id: action.id,
    label: action.label,
    icon: action.icon,
    disabled: action.disabled,
    destructive: action.destructive,
    separatorBefore: action.separatorBefore,
    title: action.title,
    onSelect: action.onSelect,
  }));

const CompactCount = ({ children }: { children: ReactNode }): JSX.Element => (
  <span className="font-mono text-micro text-muted-foreground">{children}</span>
);

const buildProjectNodes = (
  snapshot: WorkspaceSnapshot,
  actions: ProjectTreeActions,
  searchQuery: string,
  projectsOverride?: ProjectSummary[],
): TreeNode[] => {
  const lowerQuery = searchQuery.toLowerCase().trim();

  const hierarchy = (projectsOverride ?? snapshot.projects).map((project) => ({
    ...project,
    experiments: snapshot.experiments
      .filter((experiment) => experiment.projectId === project.id)
      .map((experiment) => ({
        ...experiment,
        runs: snapshot.runs.filter((run) => run.experimentId === experiment.id),
      })),
  }));

  const filtered = hierarchy.filter((project) => {
    if (!lowerQuery) return true;
    return (
      project.name.toLowerCase().includes(lowerQuery) ||
      project.summary.toLowerCase().includes(lowerQuery) ||
      project.experiments.some(
        (experiment) =>
          experiment.name.toLowerCase().includes(lowerQuery) ||
          experiment.summary.toLowerCase().includes(lowerQuery) ||
          experiment.runs.some(
            (run) =>
              run.name.toLowerCase().includes(lowerQuery) ||
              run.id.toLowerCase().includes(lowerQuery) ||
              run.summary.toLowerCase().includes(lowerQuery),
          ),
      )
    );
  });

  return filtered.map((project) => {
    const projectExpanded =
      actions.isProjectExpanded?.(project.id) ?? project.experiments.length > 0;
    const expCount = projectExpanded
      ? project.experiments.length
      : (project.experimentCount ?? project.experiments.length);
    return {
      id: project.id,
      label: project.name,
      labelClassName: statusTextClass(project.status),
      icon: Blocks,
      iconClassName: "text-muted-foreground",
      right: (
        <CompactCount>
          {projectExpanded || project.experimentCount != null ? countLabel(expCount, "exp") : "…"}
        </CompactCount>
      ),
      onSelect: () => {
        actions.onExpandProject?.(project.id);
        actions.onSelect({ objectType: "project", objectId: project.id });
      },
      actions: [
        {
          id: "open",
          label: "Open project",
          icon: ExternalLink,
          onSelect: () => {
            actions.onExpandProject?.(project.id);
            actions.onSelect({ objectType: "project", objectId: project.id });
          },
        },
        gateTreeWrite(
          {
            id: "new-experiment",
            label: "New experiment",
            icon: FlaskConical,
            onSelect: () => actions.onCreateExperiment(project.id),
          },
          actions.writeDeniedReason,
        ),
        {
          id: "refresh",
          label: "Refresh",
          icon: RefreshCw,
          onSelect: actions.onRefresh,
        },
        gateTreeWrite(
          {
            id: "delete",
            label: "Delete project",
            icon: Ban,
            destructive: true,
            separatorBefore: true,
            onSelect: () => actions.onDeleteProject(project.id),
          },
          actions.writeDeniedReason,
        ),
      ],
      // Always an array so the chevron shows; empty until expand loads experiments.
      emptyChildLabel: projectExpanded ? EMPTY_COPY.entries.title : "Loading…",
      children: project.experiments.map((experiment) => {
        const dataLoaded =
          actions.isExperimentExpanded?.(project.id, experiment.id) ?? experiment.runs.length > 0;
        const runCount = dataLoaded
          ? experiment.runs.length
          : (experiment.runCount ?? experiment.runs.length);
        const hasRunCount = dataLoaded || experiment.runCount != null;
        return {
          id: experiment.id,
          label: experiment.name,
          labelClassName: statusTextClass(experiment.status),
          icon: FlaskConical,
          iconClassName: "text-muted-foreground",
          right: <CompactCount>{hasRunCount ? countLabel(runCount, "run") : "…"}</CompactCount>,
          onSelect: () => {
            actions.onExpandProject?.(project.id);
            actions.onExpandExperiment?.(project.id, experiment.id);
            actions.onSelect({ objectType: "experiment", objectId: experiment.id });
          },
          actions: [
            {
              id: "open",
              label: "Open experiment",
              icon: ExternalLink,
              onSelect: () => {
                actions.onExpandExperiment?.(project.id, experiment.id);
                actions.onSelect({ objectType: "experiment", objectId: experiment.id });
              },
            },
            gateTreeWrite(
              {
                id: "new-run",
                label: "New run",
                icon: PlayCircle,
                onSelect: () => actions.onCreateRun(experiment.id),
              },
              actions.writeDeniedReason,
            ),
            {
              id: "open-workflow",
              label: "Open workflow",
              icon: Workflow,
              onSelect: () => {
                const workflow = snapshot.workflows.find(
                  (item) => item.experimentId === experiment.id,
                );
                if (workflow) {
                  actions.onSelect({
                    objectType: "workflow",
                    objectId: workflow.id,
                    workflowId: workflow.id,
                  });
                }
              },
              disabled: !snapshot.workflows.some((item) => item.experimentId === experiment.id),
            },
            gateTreeWrite(
              {
                id: "delete",
                label: "Delete experiment",
                icon: Ban,
                destructive: true,
                separatorBefore: true,
                onSelect: () => actions.onDeleteExperiment(experiment),
              },
              actions.writeDeniedReason,
            ),
          ],
          // Tree open + data not loaded yet → "Loading…"; loaded empty → "No runs".
          emptyChildLabel: dataLoaded ? EMPTY_COPY.runs.title : "Loading…",
          children: experiment.runs.map((run) => ({
            id: run.id,
            label: run.name || run.id,
            labelClassName: statusTextClass(run.status),
            icon: PlayCircle,
            iconClassName: "text-muted-foreground",
            onSelect: () => actions.onOpenRunView(run),
            actions: buildRunActions(run, actions),
          })),
        };
      }),
    };
  });
};

// A small chip describing a served workspace's kind/state in the nav header.
const workspaceBadge = (ws: ServedWorkspaceSummary): ReactNode => {
  const tone = ws.unreachable
    ? "bg-status-failed-soft text-status-failed-foreground"
    : ws.isRemote
      ? "bg-status-warning-soft text-status-warning-foreground"
      : "bg-muted text-muted-foreground";
  const text = ws.unreachable ? "unreachable" : ws.isRemote ? "remote" : "local";
  return <span className={`rounded-control px-2 py-1 text-micro font-medium ${tone}`}>{text}</span>;
};

// Shallow project leaves for a NON-active workspace — clicking one activates
// that workspace so its full tree loads on the next poll. Kept id-prefixed by
// workspace key so expansion/keys never collide with the active group, whose
// project ids are the real (unprefixed) ones.
const buildShallowProjectNodes = (
  projects: ProjectSummary[],
  searchQuery: string,
  workspaceKey: string,
  onActivate: () => void,
): TreeNode[] => {
  const lowerQuery = searchQuery.toLowerCase().trim();
  return projects
    .filter((project) => !lowerQuery || project.name.toLowerCase().includes(lowerQuery))
    .map((project) => ({
      id: `${workspaceKey}/${project.id}`,
      label: project.name,
      labelClassName: statusTextClass(project.status),
      icon: Blocks,
      iconClassName: "text-muted-foreground/50",
      right: <CompactCount>switch</CompactCount>,
      onSelect: onActivate,
    }));
};

// Multi-workspace nav: one collapsible header per served workspace (label +
// local/remote/unreachable badge). The ACTIVE workspace shows its full
// interactive project tree (experiments/runs); the others list project names
// that activate the workspace on click. Single-workspace callers use
// buildProjectNodes directly (unchanged flat list).
const buildWorkspaceGroupedNodes = (
  snapshot: WorkspaceSnapshot,
  actions: ProjectTreeActions,
  searchQuery: string,
  onActivateWorkspace: (ws: ServedWorkspaceSummary) => void,
): TreeNode[] => {
  return snapshot.workspaces.map((ws) => {
    // Projects without a workspaceKey (legacy/single-ws payloads) belong to
    // the active workspace — never drop them as "no projects".
    const wsProjects = snapshot.projects.filter(
      (project) => project.workspaceKey === ws.key || (project.workspaceKey == null && ws.active),
    );
    const header: TreeNode = {
      id: `ws:${ws.key}`,
      label: ws.label,
      icon: ws.unreachable ? CloudOff : ws.isRemote ? Server : HardDrive,
      iconClassName: ws.unreachable
        ? "text-status-failed-foreground"
        : ws.active
          ? "text-accent"
          : "text-muted-foreground",
      right: workspaceBadge(ws),
      emptyChildLabel: ws.unreachable ? "Unreachable" : "No projects",
    };
    if (ws.unreachable) {
      return { ...header, children: [] };
    }
    if (ws.active) {
      return { ...header, children: buildProjectNodes(snapshot, actions, searchQuery, wsProjects) };
    }
    return {
      ...header,
      onSelect: () => onActivateWorkspace(ws),
      children: buildShallowProjectNodes(wsProjects, searchQuery, ws.key, () =>
        onActivateWorkspace(ws),
      ),
    };
  });
};

interface WorkspaceSemantic {
  type: "project" | "experiment" | "run" | "asset";
  id: string;
  icon: ComponentType<{ className?: string }>;
  iconClass: string;
}

interface WorkspaceTreeActions {
  onSelect: (selection: Selection) => void;
  onCreateDirectory: (path: string) => void;
  onCreateFile: (path: string) => void;
  onCopyText: (text: string) => void;
  pathContext: PathDisplayContext;
  onRefresh: () => void;
}

const detectWorkspaceSemantic = (
  path: string,
  snapshot: WorkspaceSnapshot,
): WorkspaceSemantic | null => {
  const project = snapshot.projects.find((p) => path.endsWith(`projects/${p.id}`));
  if (project) {
    return { type: "project", id: project.id, icon: Blocks, iconClass: "text-muted-foreground" };
  }

  const experiment = snapshot.experiments.find((e) => path.endsWith(`experiments/${e.id}`));
  if (experiment) {
    return {
      type: "experiment",
      id: experiment.id,
      icon: FlaskConical,
      iconClass: "text-muted-foreground",
    };
  }

  const run = snapshot.runs.find((r) => path.endsWith(`runs/${r.id}`));
  if (run) {
    return { type: "run", id: run.id, icon: PlayCircle, iconClass: "text-muted-foreground" };
  }

  const parts = path.split("/");
  const folderName = parts[parts.length - 1];
  const parentName = parts.length > 1 ? parts[parts.length - 2] : null;
  if (parentName === "assets") {
    const asset = snapshot.assets.find((a) => a.id === folderName);
    if (asset) {
      return { type: "asset", id: asset.id, icon: Archive, iconClass: "text-muted-foreground" };
    }
  }

  return null;
};

const buildWorkspaceNodes = (
  snapshot: WorkspaceSnapshot,
  actions: WorkspaceTreeActions,
): TreeNode[] => {
  const root = snapshot.workspaceRoot;
  if (!root) return [];

  const walk = (node: WorkspaceTreeNode): TreeNode => {
    const isFile = node.kind === "file";
    const semantic = isFile ? null : detectWorkspaceSemantic(node.path, snapshot);

    const icon = semantic?.icon ?? (isFile ? FileText : Folder);
    const iconClass = semantic?.iconClass ?? (isFile ? "text-muted-foreground" : "text-foreground");

    return {
      id: node.id,
      label: node.name,
      icon,
      iconClassName: iconClass,
      meta: semantic ? (
        <span className="uppercase tracking-tighter opacity-50 group-hover:opacity-100">
          {semantic.type.substring(0, 3)}
        </span>
      ) : undefined,
      onSelect: () => {
        if (isFile) {
          actions.onSelect({
            objectType: "workspace-file",
            objectId: node.path,
            filePath: node.path,
            fileKind: detectFileKind(node.path),
            assetId: node.assetId ?? undefined,
            hasPreviewSidecar: node.hasPreviewSidecar ?? undefined,
          });
          return;
        }
        if (semantic) {
          actions.onSelect({ objectType: semantic.type, objectId: semantic.id });
        }
      },
      actions: isFile
        ? [
            {
              id: "open",
              label: "Open file",
              icon: ExternalLink,
              onSelect: () =>
                actions.onSelect({
                  objectType: "workspace-file",
                  objectId: node.path,
                  filePath: node.path,
                  fileKind: detectFileKind(node.path),
                  assetId: node.assetId ?? undefined,
                  hasPreviewSidecar: node.hasPreviewSidecar ?? undefined,
                }),
            },
            {
              id: "copy-path",
              label: "Copy path",
              icon: Copy,
              onSelect: () =>
                actions.onCopyText(formatQualifiedPath(node.path, actions.pathContext)),
            },
          ]
        : [
            ...(semantic
              ? [
                  {
                    id: "open",
                    label: `Open ${semantic.type}`,
                    icon: ExternalLink,
                    onSelect: () =>
                      actions.onSelect({ objectType: semantic.type, objectId: semantic.id }),
                  } satisfies TreeNodeAction,
                ]
              : []),
            {
              id: "new-file",
              label: "New file here",
              icon: FilePlus,
              onSelect: () => actions.onCreateFile(node.path),
            },
            {
              id: "new-folder",
              label: "New folder here",
              icon: FolderPlus,
              onSelect: () => actions.onCreateDirectory(node.path),
            },
            {
              id: "copy-path",
              label: "Copy path",
              icon: Copy,
              onSelect: () =>
                actions.onCopyText(formatQualifiedPath(node.path, actions.pathContext)),
            },
            {
              id: "refresh",
              label: "Refresh",
              icon: RefreshCw,
              separatorBefore: true,
              onSelect: actions.onRefresh,
            },
          ],
      // Always attach children array for directories so the chevron shows even
      // when childrenLoaded is false (lazy WorkspaceFs expand).
      children: isFile ? undefined : node.children.map(walk),
      emptyChildLabel: !isFile
        ? node.childrenLoaded === false
          ? "…"
          : EMPTY_COPY.emptyFolder.title
        : undefined,
    };
  };

  return [walk(root)];
};

const filterBySearch = <T extends { name: string; summary?: string }>(
  items: T[],
  searchQuery: string,
): T[] => {
  if (!searchQuery) return items;
  const lower = searchQuery.toLowerCase();
  return items.filter(
    (item) =>
      item.name.toLowerCase().includes(lower) || item.summary?.toLowerCase().includes(lower),
  );
};

// Group assets under their owning scope: Project → Experiment → Run. An asset's
// scope chain comes from its `scope_ids` (projectId / experimentId / runId);
// assets with no project fall under a "Workspace" group. Container nodes show
// the asset count and open their entity; leaves open the asset.
const buildAssetNodes = (
  snapshot: WorkspaceSnapshot,
  onSelect: (selection: Selection) => void,
  onCopyText: (text: string) => void,
  searchQuery: string,
): TreeNode[] => {
  // Dedup by id (the catalog + per-project fetches can overlap).
  const byId = new Map<string, AssetSummary>();
  for (const asset of filterBySearch(snapshot.assets, searchQuery)) {
    if (!byId.has(asset.id)) byId.set(asset.id, asset);
  }
  const assets = [...byId.values()];

  const projName = (id: string): string => snapshot.projects.find((p) => p.id === id)?.name ?? id;
  const expName = (id: string): string => snapshot.experiments.find((e) => e.id === id)?.name ?? id;
  const runName = (id: string): string => snapshot.runs.find((r) => r.id === id)?.name ?? id;

  const assetLeaf = (asset: AssetSummary): TreeNode => ({
    id: asset.id,
    label: asset.name,
    icon: Archive,
    iconClassName: "text-muted-foreground",
    right: <StatusBadge status={asset.status} size="sm" />,
    onSelect: () => onSelect({ objectType: "asset", objectId: asset.id }),
    actions: [
      {
        id: "open",
        label: "Open asset",
        icon: ExternalLink,
        onSelect: () => onSelect({ objectType: "asset", objectId: asset.id }),
      },
      {
        id: "copy-id",
        label: "Copy asset ID",
        icon: Copy,
        onSelect: () => onCopyText(asset.id),
      },
    ],
  });

  // Bucket by (project, experiment, run); a missing id means the scope stops there.
  const groupBy = <T,>(rows: AssetSummary[], key: (a: AssetSummary) => T | undefined) => {
    const direct: AssetSummary[] = [];
    const groups = new Map<T, AssetSummary[]>();
    for (const a of rows) {
      const k = key(a);
      if (k === undefined) direct.push(a);
      else groups.set(k, [...(groups.get(k) ?? []), a]);
    }
    return { direct, groups };
  };
  const byLabel = (a: TreeNode, b: TreeNode): number => a.label.localeCompare(b.label);

  const { direct: workspaceAssets, groups: byProject } = groupBy(assets, (a) => a.projectId);

  const projectNodes: TreeNode[] = [...byProject.entries()]
    .map(([projectId, projAssets]): TreeNode => {
      const { direct: projDirect, groups: byExp } = groupBy(projAssets, (a) => a.experimentId);
      const expNodes: TreeNode[] = [...byExp.entries()]
        .map(([expId, expAssets]): TreeNode => {
          const { direct: expDirect, groups: byRun } = groupBy(expAssets, (a) => a.runId);
          const runNodes: TreeNode[] = [...byRun.entries()]
            .map(
              ([runId, runAssets]): TreeNode => ({
                id: `asset-run-${runId}`,
                label: runName(runId),
                icon: PlayCircle,
                iconClassName: "text-muted-foreground",
                right: <CompactCount>{runAssets.length}</CompactCount>,
                onSelect: () => onSelect({ objectType: "run", objectId: runId }),
                children: [...runAssets]
                  .sort((a, b) => a.name.localeCompare(b.name))
                  .map(assetLeaf),
              }),
            )
            .sort(byLabel);
          return {
            id: `asset-exp-${expId}`,
            label: expName(expId),
            icon: FlaskConical,
            iconClassName: "text-muted-foreground",
            right: <CompactCount>{expAssets.length}</CompactCount>,
            onSelect: () => onSelect({ objectType: "experiment", objectId: expId }),
            children: [...expDirect.map(assetLeaf), ...runNodes],
          };
        })
        .sort(byLabel);
      return {
        id: `asset-proj-${projectId}`,
        label: projName(projectId),
        icon: Blocks,
        iconClassName: "text-muted-foreground",
        right: <CompactCount>{projAssets.length}</CompactCount>,
        onSelect: () => onSelect({ objectType: "project", objectId: projectId }),
        children: [...projDirect.map(assetLeaf), ...expNodes],
      };
    })
    .sort(byLabel);

  if (workspaceAssets.length > 0) {
    projectNodes.push({
      id: "asset-workspace",
      label: "Workspace",
      icon: FolderTree,
      iconClassName: "text-muted-foreground",
      right: <CompactCount>{workspaceAssets.length}</CompactCount>,
      children: workspaceAssets.map(assetLeaf),
    });
  }
  return projectNodes;
};

const buildWorkflowNodes = (
  snapshot: WorkspaceSnapshot,
  onSelect: (selection: Selection) => void,
  onCopyText: (text: string) => void,
  searchQuery: string,
): TreeNode[] => {
  return filterBySearch(snapshot.workflows, searchQuery).map((workflow) => ({
    id: workflow.id,
    label: workflow.name,
    icon: Workflow,
    iconClassName: "text-muted-foreground",
    right: <StatusBadge status={workflow.status} size="sm" />,
    onSelect: () =>
      onSelect({ objectType: "workflow", objectId: workflow.id, workflowId: workflow.id }),
    actions: [
      {
        id: "open",
        label: "Open workflow",
        icon: ExternalLink,
        onSelect: () =>
          onSelect({ objectType: "workflow", objectId: workflow.id, workflowId: workflow.id }),
      },
      {
        id: "open-experiment",
        label: "Open experiment",
        icon: FlaskConical,
        onSelect: () => onSelect({ objectType: "experiment", objectId: workflow.experimentId }),
      },
      {
        id: "copy-id",
        label: "Copy workflow ID",
        icon: Copy,
        onSelect: () => onCopyText(workflow.id),
      },
    ],
  }));
};

// Sidebar rows are narrow; show only the first sentence/clause of the
// markdown-stripped task title so the StatusBadge stays visible. Full
// text is on the row tooltip and the entity header inside the task view.
const shortenTaskTitle = (session: AgentSessionSummary): string => {
  const clean = agentTaskDisplayTitle(session, 200);
  const sentenceEnd = clean.search(/[.!?。！？]/);
  const clipped = sentenceEnd > 0 ? clean.slice(0, sentenceEnd) : clean;
  return clipped.length > 32 ? `${clipped.slice(0, 30).trim()}…` : clipped;
};

const buildAgentNodes = (
  snapshot: WorkspaceSnapshot,
  onSelect: (selection: Selection) => void,
  onCopyText: (text: string) => void,
  onDeleteAgent: (session: AgentSessionSummary) => void,
): TreeNode[] => {
  return snapshot.agentSessions.map((session) => {
    return {
      id: session.id,
      label: shortenTaskTitle(session),
      hoverTitle: session.goal,
      icon: Bot,
      iconClassName: "text-muted-foreground",
      right: <StatusBadge status={session.status} size="sm" dot showLabel={false} />,
      onSelect: () => onSelect({ objectType: "agent", objectId: session.id }),
      actions: [
        {
          id: "open",
          label: "Open task",
          icon: ExternalLink,
          onSelect: () => onSelect({ objectType: "agent", objectId: session.id }),
        },
        {
          id: "copy-id",
          label: "Copy task ID",
          icon: Copy,
          onSelect: () => onCopyText(session.id),
        },
        {
          id: "delete",
          label: "Delete task",
          icon: Trash2,
          destructive: true,
          separatorBefore: true,
          onSelect: () => onDeleteAgent(session),
        },
      ],
    };
  });
};

const buildProjectExpandPath = (
  snapshot: WorkspaceSnapshot,
  activeId: string | undefined,
  searchQuery: string,
): string[] => {
  if (searchQuery) {
    const ids: string[] = [];
    for (const project of snapshot.projects) {
      ids.push(project.id);
      for (const experiment of snapshot.experiments.filter((e) => e.projectId === project.id)) {
        ids.push(experiment.id);
      }
    }
    return ids;
  }

  if (!activeId) return [];

  const ids: string[] = [];
  if (snapshot.projects.some((p) => p.id === activeId)) {
    ids.push(activeId);
  }
  const experiment = snapshot.experiments.find((e) => e.id === activeId);
  if (experiment) {
    ids.push(experiment.projectId, experiment.id);
  }
  const run = snapshot.runs.find((r) => r.id === activeId);
  if (run) {
    ids.push(run.projectId, run.experimentId);
  }
  return ids;
};

export const LeftPanel = ({
  view,
  selection,
  snapshot,
  onViewChange,
  onSelect,
  onOpenWorkspace,
  onCreateDirectory,
  onCreateFile,
  onRefresh,
  onExpandDirectory,
  onExpandProject,
  onExpandExperiment,
  isProjectExpanded,
  isExperimentExpanded,
  dataEpoch = 0,
  searchQuery = "",
}: LeftPanelProps): JSX.Element => {
  const listHeader = listHeaderByView[view];
  const hasWorkspace = Boolean(snapshot.workspaceRoot);
  // Active served workspace — always surface its identity so a single remote
  // mount (e.g. ``Arrhenius:/home/…/mace-nve``) is never anonymous under
  // "Projects". Multi-ws tree headers still group by workspace; this is the
  // strip above the list for every projects view.
  const activeWorkspace =
    snapshot.workspaces.find((w) => w.active) ?? snapshot.workspaces[0] ?? null;
  const [createExperimentProjectId, setCreateExperimentProjectId] = useState<string | null>(null);
  const [createRunExperimentId, setCreateRunExperimentId] = useState<string | null>(null);
  const { prompt, dialog: promptDialog } = usePrompt();
  const { confirm, dialog: confirmDialog } = useConfirm();
  const { alert, dialog: alertDialog } = useAlert();
  const [searchParams, setSearchParams] = useSearchParams();
  const runsFilters = useMemo<WorkspaceRunsFilters>(
    () => (view === "runs" ? parseFilterParams(searchParams) : {}),
    [searchParams, view],
  );
  // Only subscribe to the runs poller when the user is actually looking at
  // the runs view; otherwise the LeftPanel would keep hitting /api/runs in
  // the background even on the workspace/projects views.
  const { rows: runsRows } = useWorkspaceRuns({ enabled: view === "runs" });
  const runsFacets = useMemo(
    () => computeFacetCounts(runsRows, runsFilters),
    [runsRows, runsFilters],
  );
  const handleRunsFiltersChange = (next: WorkspaceRunsFilters): void => {
    setSearchParams((prev) => writeFilterParams(prev, next), { replace: true });
  };

  const prevSelectionRef = useRef(selection);
  useEffect(() => {
    const prev = prevSelectionRef.current;
    const isSame =
      prev === selection ||
      (prev?.objectId === selection?.objectId && prev?.objectType === selection?.objectType);
    if (isSame) return;
    prevSelectionRef.current = selection;
  }, [selection]);

  const activeId = selection ? selection.objectId : undefined;

  const handleOpenWorkspace = async (): Promise<void> => {
    const path = await prompt({
      title: "Open workspace",
      label: "Workspace path",
      placeholder: "/path/to/workspace",
      confirmLabel: "Open",
    });
    if (!path) return;
    try {
      await onOpenWorkspace(path);
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) {
        const create = await confirm({
          title: "Create workspace?",
          description: `${path} does not exist.`,
          confirmLabel: "Create",
        });
        if (!create) return;
        try {
          await onOpenWorkspace(path, { createIfMissing: true });
        } catch (retryError) {
          await alert({ title: "Open failed", description: errorDetail(retryError) });
        }
        return;
      }
      await alert({ title: "Open failed", description: errorDetail(error) });
    }
  };
  const handleCreateFile = async (): Promise<void> => {
    const path = await prompt({
      title: "New file",
      label: "File path",
      description: "Relative to the workspace root.",
      placeholder: "notebooks/example.md",
      confirmLabel: "Create",
    });
    if (!path) return;
    onCreateFile(path);
  };
  const handleCreateDirectory = async (): Promise<void> => {
    const path = await prompt({
      title: "New folder",
      label: "Folder path",
      description: "Relative to the workspace root.",
      placeholder: "experiments/new",
      confirmLabel: "Create",
    });
    if (!path) return;
    onCreateDirectory(path);
  };
  const handleCreateFileInDirectory = async (directoryPath: string): Promise<void> => {
    const name = await prompt({
      title: "New file",
      label: "File name",
      description: directoryPath,
      confirmLabel: "Create",
    });
    if (!name) return;
    onCreateFile(joinWorkspacePath(directoryPath, name));
  };
  const handleCreateDirectoryInDirectory = async (directoryPath: string): Promise<void> => {
    const name = await prompt({
      title: "New folder",
      label: "Folder name",
      description: directoryPath,
      confirmLabel: "Create",
    });
    if (!name) return;
    onCreateDirectory(joinWorkspacePath(directoryPath, name));
  };
  const handleCopyText = (text: string): void => {
    void copyText(text);
  };
  const handleOpenRunView = (run: RunSummary, objectView?: ObjectView): void => {
    onSelect({ objectType: "run", objectId: run.id, objectView });
  };
  const handleDeleteProject = async (projectId: string): Promise<void> => {
    const confirmed = await confirm({
      title: "Delete project?",
      description: (
        <>
          Project{" "}
          <InlineCode className="rounded-control bg-muted px-1 py-1 text-label">
            {projectId}
          </InlineCode>{" "}
          and its experiments will be removed from the workspace.
        </>
      ),
      confirmLabel: "Delete",
      destructive: true,
    });
    if (!confirmed) return;
    try {
      await workspaceApi.deleteProject(projectId);
      onRefresh();
    } catch (error) {
      console.error("Failed to delete project:", error);
      void alert({
        title: "Failed to delete project",
        description: error instanceof Error ? error.message : String(error),
      });
    }
  };
  const handleDeleteExperiment = async (experiment: ExperimentSummary): Promise<void> => {
    const confirmed = await confirm({
      title: "Delete experiment?",
      description: (
        <>
          Experiment{" "}
          <InlineCode className="rounded-control bg-muted px-1 py-1 text-label">
            {experiment.id}
          </InlineCode>{" "}
          and its runs will be removed.
        </>
      ),
      confirmLabel: "Delete",
      destructive: true,
    });
    if (!confirmed) return;
    try {
      await workspaceApi.deleteExperiment(experiment.projectId, experiment.id);
      onRefresh();
    } catch (error) {
      console.error("Failed to delete experiment:", error);
      void alert({
        title: "Failed to delete experiment",
        description: error instanceof Error ? error.message : String(error),
      });
    }
  };
  const handleDeleteAgentTask = async (session: AgentSessionSummary): Promise<void> => {
    const confirmed = await confirm({
      title: "Delete agent task?",
      description: (
        <>
          Agent task{" "}
          <InlineCode className="rounded-control bg-muted px-1 py-1 text-label">
            {session.id}
          </InlineCode>{" "}
          will be removed from the task list. If it is running, its current turn will be cancelled.
        </>
      ),
      confirmLabel: "Delete",
      destructive: true,
    });
    if (!confirmed) return;
    try {
      await agentApi.deleteSession(session.id);
      if (selection?.objectType === "agent" && selection.objectId === session.id) {
        onSelect({ objectType: "agent", objectId: "new" });
      }
      onRefresh();
    } catch (error) {
      console.error("Failed to delete agent task:", error);
      void alert({
        title: "Failed to delete agent task",
        description: error instanceof Error ? error.message : String(error),
      });
    }
  };

  const { writeDeniedReason } = usePermissions();

  const pathContext: PathDisplayContext = useMemo(() => {
    const active = snapshot.workspaces.find((w) => w.active) ?? snapshot.workspaces[0] ?? null;
    return {
      root: getWorkspaceFs().root,
      workspace: active
        ? { label: active.label, isRemote: active.isRemote, path: active.path }
        : null,
    };
  }, [snapshot.workspaces]);

  const projectTreeActions: ProjectTreeActions = {
    onSelect,
    onCreateExperiment: setCreateExperimentProjectId,
    onCreateRun: setCreateRunExperimentId,
    writeDeniedReason,
    onDeleteProject: (projectId) => {
      void handleDeleteProject(projectId);
    },
    onDeleteExperiment: (experiment) => {
      void handleDeleteExperiment(experiment);
    },
    onOpenRunView: handleOpenRunView,
    onCopyText: handleCopyText,
    pathContext,
    onRefresh,
    onExpandProject,
    onExpandExperiment,
    isProjectExpanded,
    isExperimentExpanded,
  };

  const workspaceTreeActions: WorkspaceTreeActions = {
    onSelect,
    onCreateDirectory: (path) => {
      void handleCreateDirectoryInDirectory(path);
    },
    onCreateFile: (path) => {
      void handleCreateFileInDirectory(path);
    },
    onCopyText: handleCopyText,
    pathContext,
    onRefresh,
  };

  const handleActivateWorkspace = (ws: ServedWorkspaceSummary): void => {
    void workspaceApi
      .activateServedWorkspace(ws)
      .then(() => onRefresh())
      .catch((err) => console.warn(`Failed to switch to workspace ${ws.key}:`, err));
  };

  // >1 served workspace → group projects under per-workspace headers; otherwise
  // today's flat project list (single-workspace behaviour unchanged).
  const projectNodes =
    snapshot.workspaces.length > 1
      ? buildWorkspaceGroupedNodes(
          snapshot,
          projectTreeActions,
          searchQuery,
          handleActivateWorkspace,
        )
      : buildProjectNodes(snapshot, projectTreeActions, searchQuery);
  const workspaceNodes = buildWorkspaceNodes(snapshot, workspaceTreeActions);
  const assetNodes = buildAssetNodes(snapshot, onSelect, handleCopyText, searchQuery);
  const workflowNodes = buildWorkflowNodes(snapshot, onSelect, handleCopyText, searchQuery);
  const agentNodes = buildAgentNodes(snapshot, onSelect, handleCopyText, (session) => {
    void handleDeleteAgentTask(session);
  });

  const projectExpandPath = useMemo(
    () => buildProjectExpandPath(snapshot, activeId, searchQuery),
    [snapshot, activeId, searchQuery],
  );
  const workspaceExpandPath = useMemo(
    () => (snapshot.workspaceRoot ? [snapshot.workspaceRoot.id] : []),
    [snapshot.workspaceRoot],
  );

  const treeByView: Record<LeftPanelView, JSX.Element> = {
    projects: (
      <TreeView
        nodes={projectNodes}
        activeId={activeId}
        expandPath={projectExpandPath}
        dataEpoch={dataEpoch}
        emptyTitle={searchQuery ? EMPTY_COPY.projectsFilter.title : EMPTY_COPY.entries.title}
        onExpand={(nodeId) => {
          // Project ids live at the top level of the snapshot.
          if (snapshot.projects.some((p) => p.id === nodeId)) {
            onExpandProject?.(nodeId);
            return;
          }
          // Experiment ids — resolve parent project, then load runs.
          const experiment = snapshot.experiments.find((e) => e.id === nodeId);
          if (experiment) {
            onExpandExperiment?.(experiment.projectId, experiment.id);
          }
        }}
      />
    ),
    workspace: (
      <TreeView
        nodes={workspaceNodes}
        activeId={activeId}
        expandPath={workspaceExpandPath}
        dataEpoch={dataEpoch}
        emptyTitle={EMPTY_COPY.workspace.title}
        onExpand={(nodeId) => {
          // nodeId is the workspace path (see buildWorkspaceNodes).
          onExpandDirectory?.(nodeId);
        }}
      />
    ),
    runs: (
      <RunsFacetPanel
        facets={runsFacets}
        filters={runsFilters}
        onFiltersChange={handleRunsFiltersChange}
      />
    ),
    activity: (
      <div className="space-y-2 px-2 py-3 text-label text-muted-foreground">
        <p className="font-medium text-foreground">Event spine</p>
        <p>
          The center panel shows the workspace-wide activity timeline (runs, knowledge, assets).
          Filter by event type there.
        </p>
      </div>
    ),
    asset: <TreeView nodes={assetNodes} activeId={activeId} emptyTitle={EMPTY_COPY.assets.title} />,
    workflow: (
      <TreeView nodes={workflowNodes} activeId={activeId} emptyTitle={EMPTY_COPY.entries.title} />
    ),
    agent: (
      <TreeView
        nodes={agentNodes}
        activeId={activeId}
        emptyIcon={<Sparkles className="h-control w-control" />}
        emptyTitle={EMPTY_COPY.agentSessions.title}
        emptyDescription={EMPTY_COPY.agentSessions.description}
      />
    ),
    knowledge: <DocTree snapshot={snapshot} activeId={activeId} onSelect={onSelect} />,
    settings: (
      <nav className="space-y-1 px-1 pb-4 text-label">
        <div className="rounded-control bg-muted/30 px-2 py-2 font-medium text-foreground">
          Compute targets
        </div>
      </nav>
    ),
  };

  const createRunExperiment = createRunExperimentId
    ? snapshot.experiments.find((experiment) => experiment.id === createRunExperimentId)
    : null;

  return (
    <div className="flex h-full">
      <TooltipProvider>
        <div className="flex w-14 flex-col items-center gap-2 border-r border-border bg-muted/20 py-4">
          {viewOptions.map((option) => {
            const isActive = view === option.id;
            return (
              <Tooltip key={option.id}>
                <TooltipTrigger asChild>
                  <WorkbenchToggleAction
                    label={option.label}
                    pressed={isActive}
                    onClick={() => onViewChange(option.id)}
                  >
                    <option.icon className="h-4 w-4" />
                  </WorkbenchToggleAction>
                </TooltipTrigger>
                <TooltipContent side="right">{option.label}</TooltipContent>
              </Tooltip>
            );
          })}
          <div className="mt-auto">
            <Tooltip>
              <TooltipTrigger asChild>
                <WorkbenchToggleAction
                  label="Settings"
                  pressed={view === "settings"}
                  onClick={() => onViewChange("settings")}
                >
                  <Settings className="h-4 w-4" />
                </WorkbenchToggleAction>
              </TooltipTrigger>
              <TooltipContent side="right">Settings</TooltipContent>
            </Tooltip>
          </div>
        </div>
      </TooltipProvider>

      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <div className="space-y-1 px-4 py-3">
          <div className="flex items-center justify-between gap-2">
            <p className="text-label font-semibold uppercase tracking-wide text-muted-foreground">
              {listHeader}
            </p>

            {view === "projects" && (
              <div className="flex items-center gap-1">
                <WorkbenchIconAction
                  label="Refresh projects"
                  kind="ghost"
                  onClick={onRefresh}
                  aria-label="Refresh projects"
                >
                  <RefreshCw className="h-4 w-4" />
                </WorkbenchIconAction>
                <CreateProjectDialog
                  onProjectCreated={onRefresh}
                  writeDeniedReason={writeDeniedReason}
                />
              </div>
            )}

            {view === "workspace" && (
              <div className="flex items-center gap-1">
                {!hasWorkspace ? (
                  <WorkbenchIconAction
                    label="Open workspace"
                    kind="ghost"
                    deniedReason={writeDeniedReason}
                    onClick={() => {
                      void handleOpenWorkspace();
                    }}
                  >
                    <FolderOpen className="h-4 w-4" />
                  </WorkbenchIconAction>
                ) : (
                  <>
                    <WorkbenchIconAction
                      label="New file"
                      kind="ghost"
                      deniedReason={writeDeniedReason}
                      onClick={() => {
                        void handleCreateFile();
                      }}
                    >
                      <FilePlus className="h-4 w-4" />
                    </WorkbenchIconAction>
                    <WorkbenchIconAction
                      label="New folder"
                      kind="ghost"
                      deniedReason={writeDeniedReason}
                      onClick={() => {
                        void handleCreateDirectory();
                      }}
                    >
                      <FolderPlus className="h-4 w-4" />
                    </WorkbenchIconAction>
                    <WorkbenchIconAction
                      label="Refresh workspace"
                      kind="ghost"
                      className="h-control-compact w-control-compact"
                      onClick={onRefresh}
                      aria-label="Refresh workspace"
                    >
                      <RefreshCw className="h-4 w-4" />
                    </WorkbenchIconAction>
                  </>
                )}
              </div>
            )}

            {view === "agent" && (
              <div className="flex items-center gap-1">
                <WorkbenchIconAction
                  label="Agent settings"
                  kind="ghost"
                  onClick={() => onSelect({ objectType: "agent", objectId: "settings" })}
                  title="Agents, model, skills, tools, and MCP"
                >
                  <Settings className="h-4 w-4" />
                </WorkbenchIconAction>
                <WorkbenchIconAction
                  label="New agent task"
                  kind="ghost"
                  deniedReason={writeDeniedReason}
                  onClick={() => onSelect({ objectType: "agent", objectId: "new" })}
                >
                  <Plus className="h-4 w-4" />
                </WorkbenchIconAction>
              </div>
            )}
          </div>

          {/* Always-on workspace identity for the projects inventory so a
              single-host mount is never mistaken for "local anonymous". */}
          {view === "projects" && activeWorkspace && (
            <div className="flex min-w-0 items-center gap-1.5 pt-0.5" title={activeWorkspace.label}>
              {activeWorkspace.unreachable ? (
                <CloudOff className="h-3 w-3 flex-none text-status-failed-foreground" aria-hidden />
              ) : activeWorkspace.isRemote ? (
                <Server className="h-3 w-3 flex-none text-status-warning-foreground" aria-hidden />
              ) : (
                <HardDrive className="h-3 w-3 flex-none text-muted-foreground" aria-hidden />
              )}
              <span className="min-w-0 truncate font-mono text-micro text-foreground/80">
                {activeWorkspace.label}
              </span>
              {workspaceBadge(activeWorkspace)}
            </div>
          )}
          <Separator />
        </div>
        <ScrollArea className="flex-1 px-4 pb-4">{treeByView[view]}</ScrollArea>
      </div>
      {createExperimentProjectId && (
        <CreateExperimentDialog
          projectId={createExperimentProjectId}
          open
          trigger={null}
          onOpenChange={(nextOpen) => {
            if (!nextOpen) setCreateExperimentProjectId(null);
          }}
          onExperimentCreated={onRefresh}
        />
      )}
      {createRunExperiment && (
        <CreateRunDialog
          projectId={createRunExperiment.projectId}
          experimentId={createRunExperiment.id}
          workflowFile={createRunExperiment.workflowFile || ""}
          open
          trigger={null}
          onOpenChange={(nextOpen) => {
            if (!nextOpen) setCreateRunExperimentId(null);
          }}
          onRunCreated={(runId) => {
            onRefresh();
            setCreateRunExperimentId(null);
            onSelect({ objectType: "run", objectId: runId });
          }}
        />
      )}
      {promptDialog}
      {confirmDialog}
      {alertDialog}
    </div>
  );
};
