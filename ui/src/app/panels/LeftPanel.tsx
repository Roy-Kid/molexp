import {
  Archive,
  Ban,
  Blocks,
  Bot,
  Copy,
  ExternalLink,
  FilePlus,
  FileText,
  FlaskConical,
  Folder,
  FolderOpen,
  FolderPlus,
  FolderTree,
  PlayCircle,
  Plus,
  RefreshCw,
  Settings,
  Sparkles,
  Terminal,
  Workflow,
} from "lucide-react";
import type { ComponentType, SVGProps } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { CreateExperimentDialog } from "@/app/components/CreateExperimentDialog";
import { CreateProjectDialog } from "@/app/components/CreateProjectDialog";
import { CreateRunDialog } from "@/app/components/CreateRunDialog";
import { EMPTY_COPY, StatusBadge } from "@/app/components/entity";
import type { TreeNode, TreeNodeAction } from "@/app/panels/TreeView";
import { TreeView } from "@/app/panels/TreeView";
import { computeFacetCounts } from "@/app/runs/aggregates";
import { parseFilterParams, writeFilterParams } from "@/app/runs/filterParams";
import { RunsFacetPanel } from "@/app/runs/RunsFacetPanel";
import type { WorkspaceRunsFilters } from "@/app/runs/types";
import { useWorkspaceRuns } from "@/app/runs/useWorkspaceRuns";
import { workspaceApi } from "@/app/state/api";
import type {
  ExperimentSummary,
  FileKind,
  LeftPanelView,
  ObjectView,
  RunSummary,
  Selection,
  SemanticStatus,
  WorkspaceSnapshot,
  WorkspaceTreeNode,
} from "@/app/types";
import { useAlert, useConfirm } from "@/components/ConfirmDialog";
import { usePrompt } from "@/components/PromptDialog";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface LeftPanelProps {
  view: LeftPanelView;
  selection: Selection | null;
  snapshot: WorkspaceSnapshot;
  searchQuery?: string;
  onViewChange: (view: LeftPanelView) => void;
  onSelect: (selection: Selection) => void;
  onOpenWorkspace: (path: string) => void;
  onCreateDirectory: (path: string) => void;
  onCreateFile: (path: string) => void;
  onRefresh: () => void;
}

interface ViewOption {
  id: LeftPanelView;
  label: string;
  icon: ComponentType<SVGProps<SVGSVGElement>>;
}

const viewOptions: ViewOption[] = [
  { id: "projects", label: "Projects", icon: Blocks },
  { id: "workspace", label: "Workspace", icon: FolderTree },
  { id: "runs", label: "Runs", icon: PlayCircle },
  { id: "asset", label: "Asset", icon: Archive },
  { id: "workflow", label: "Workflow", icon: Workflow },
  { id: "agent", label: "Agent Tasks", icon: Bot },
];

const listHeaderByView: Record<LeftPanelView, string> = {
  projects: "Projects",
  workspace: "Workspace",
  runs: "Runs",
  asset: "Assets",
  workflow: "Workflows",
  agent: "Agent Tasks",
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

const runIconClass = (status: SemanticStatus): string => {
  switch (status) {
    case "succeeded":
      return "text-emerald-500";
    case "failed":
      return "text-rose-500";
    case "running":
      return "text-blue-500";
    default:
      return "text-muted-foreground";
  }
};

const terminalRunStatuses = new Set<SemanticStatus>([
  "succeeded",
  "failed",
  "cancelled",
  "skipped",
]);

const joinWorkspacePath = (parent: string, child: string): string => {
  const trimmedChild = child.trim();
  if (!trimmedChild) return parent;
  if (trimmedChild.startsWith("/")) return trimmedChild;
  return `${parent.replace(/\/$/, "")}/${trimmedChild}`;
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
  onCancelRun: (run: RunSummary) => void;
  onOpenRunView: (run: RunSummary, view?: ObjectView) => void;
  onCopyText: (text: string) => void;
  onRefresh: () => void;
}

const buildRunActions = (run: RunSummary, actions: ProjectTreeActions): TreeNodeAction[] => [
  {
    id: "open",
    label: "Open run",
    icon: ExternalLink,
    onSelect: () => actions.onOpenRunView(run),
  },
  {
    id: "logs",
    label: "View logs",
    icon: Terminal,
    onSelect: () => actions.onOpenRunView(run, "logs"),
  },
  {
    id: "snapshot",
    label: "View snapshot",
    icon: Archive,
    onSelect: () => actions.onOpenRunView(run, "snapshot"),
  },
  {
    id: "copy-id",
    label: "Copy run ID",
    icon: Copy,
    onSelect: () => actions.onCopyText(run.id),
  },
  {
    id: "cancel",
    label: "Mark cancelled",
    icon: Ban,
    disabled: terminalRunStatuses.has(run.status),
    destructive: true,
    separatorBefore: true,
    title: terminalRunStatuses.has(run.status)
      ? "Terminal runs cannot be cancelled."
      : "Updates workspace status only; it does not cancel a scheduler job.",
    onSelect: () => actions.onCancelRun(run),
  },
];

const buildProjectNodes = (
  snapshot: WorkspaceSnapshot,
  actions: ProjectTreeActions,
  searchQuery: string,
): TreeNode[] => {
  const lowerQuery = searchQuery.toLowerCase().trim();

  const hierarchy = snapshot.projects.map((project) => ({
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

  return filtered.map((project) => ({
    id: project.id,
    label: project.name,
    icon: Blocks,
    iconClassName: "text-blue-500",
    meta: `${project.experiments.length} exp`,
    onSelect: () => actions.onSelect({ objectType: "project", objectId: project.id }),
    actions: [
      {
        id: "open",
        label: "Open project",
        icon: ExternalLink,
        onSelect: () => actions.onSelect({ objectType: "project", objectId: project.id }),
      },
      {
        id: "new-experiment",
        label: "New experiment",
        icon: FlaskConical,
        onSelect: () => actions.onCreateExperiment(project.id),
      },
      {
        id: "refresh",
        label: "Refresh",
        icon: RefreshCw,
        onSelect: actions.onRefresh,
      },
      {
        id: "delete",
        label: "Delete project",
        icon: Ban,
        destructive: true,
        separatorBefore: true,
        onSelect: () => actions.onDeleteProject(project.id),
      },
    ],
    children: project.experiments.map((experiment) => ({
      id: experiment.id,
      label: experiment.name,
      icon: FlaskConical,
      iconClassName: "text-purple-500",
      right: <StatusBadge status={experiment.status} size="sm" />,
      meta: `${experiment.runs.length} runs`,
      onSelect: () => actions.onSelect({ objectType: "experiment", objectId: experiment.id }),
      actions: [
        {
          id: "open",
          label: "Open experiment",
          icon: ExternalLink,
          onSelect: () => actions.onSelect({ objectType: "experiment", objectId: experiment.id }),
        },
        {
          id: "new-run",
          label: "New run",
          icon: PlayCircle,
          onSelect: () => actions.onCreateRun(experiment.id),
        },
        {
          id: "open-workflow",
          label: "Open workflow",
          icon: Workflow,
          onSelect: () => {
            const workflow = snapshot.workflows.find((item) => item.experimentId === experiment.id);
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
        {
          id: "delete",
          label: "Delete experiment",
          icon: Ban,
          destructive: true,
          separatorBefore: true,
          onSelect: () => actions.onDeleteExperiment(experiment),
        },
      ],
      emptyChildLabel: EMPTY_COPY.runs.title,
      children: experiment.runs.map((run) => ({
        id: run.id,
        label: run.name || run.id,
        icon: PlayCircle,
        iconClassName: runIconClass(run.status),
        right: <StatusBadge status={run.status} size="sm" />,
        meta: run.profile ?? run.id.substring(0, 8),
        onSelect: () => actions.onOpenRunView(run),
        actions: buildRunActions(run, actions),
      })),
    })),
  }));
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
  onRefresh: () => void;
}

const detectWorkspaceSemantic = (
  path: string,
  snapshot: WorkspaceSnapshot,
): WorkspaceSemantic | null => {
  const project = snapshot.projects.find((p) => path.endsWith(`projects/${p.id}`));
  if (project) {
    return { type: "project", id: project.id, icon: Blocks, iconClass: "text-blue-500" };
  }

  const experiment = snapshot.experiments.find((e) => path.endsWith(`experiments/${e.id}`));
  if (experiment) {
    return {
      type: "experiment",
      id: experiment.id,
      icon: FlaskConical,
      iconClass: "text-purple-500",
    };
  }

  const run = snapshot.runs.find((r) => path.endsWith(`runs/${r.id}`));
  if (run) {
    return { type: "run", id: run.id, icon: PlayCircle, iconClass: runIconClass(run.status) };
  }

  const parts = path.split("/");
  const folderName = parts[parts.length - 1];
  const parentName = parts.length > 1 ? parts[parts.length - 2] : null;
  if (parentName === "assets") {
    const asset = snapshot.assets.find((a) => a.id === folderName);
    if (asset) {
      return { type: "asset", id: asset.id, icon: Archive, iconClass: "text-amber-500" };
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
                }),
            },
            {
              id: "copy-path",
              label: "Copy path",
              icon: Copy,
              onSelect: () => actions.onCopyText(node.path),
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
              onSelect: () => actions.onCopyText(node.path),
            },
            {
              id: "refresh",
              label: "Refresh",
              icon: RefreshCw,
              separatorBefore: true,
              onSelect: actions.onRefresh,
            },
          ],
      children: isFile ? undefined : node.children.map(walk),
      emptyChildLabel: !isFile ? EMPTY_COPY.emptyFolder.title : undefined,
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

const buildAssetNodes = (
  snapshot: WorkspaceSnapshot,
  onSelect: (selection: Selection) => void,
  onCopyText: (text: string) => void,
  searchQuery: string,
): TreeNode[] => {
  return filterBySearch(snapshot.assets, searchQuery).map((asset) => ({
    id: asset.id,
    label: asset.name,
    icon: Archive,
    iconClassName: "text-amber-500",
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
  }));
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
    iconClassName: "text-sky-500",
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
// goal so the StatusBadge stays visible. Full text is on the entity
// header inside the task view.
const shortenGoal = (goal: string): string => {
  const firstLine = goal.split("\n")[0]?.trim() ?? "";
  const sentenceEnd = firstLine.search(/[.!?。！？]/);
  const clipped = sentenceEnd > 0 ? firstLine.slice(0, sentenceEnd) : firstLine;
  return clipped.length > 32 ? `${clipped.slice(0, 30).trim()}…` : clipped;
};

const buildAgentNodes = (
  snapshot: WorkspaceSnapshot,
  onSelect: (selection: Selection) => void,
  onCopyText: (text: string) => void,
): TreeNode[] => {
  return snapshot.agentSessions.map((session) => {
    const isLive = session.status === "running";
    return {
      id: session.id,
      label: shortenGoal(session.goal),
      icon: Bot,
      iconClassName: isLive ? "text-info animate-pulse" : "text-violet-500",
      right: (
        <span className="flex items-center gap-1">
          {isLive && (
            <span
              title="Live"
              className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-info"
            />
          )}
          <StatusBadge status={session.status} size="sm" dot showLabel={false} />
        </span>
      ),
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
  searchQuery = "",
}: LeftPanelProps): JSX.Element => {
  const listHeader = listHeaderByView[view];
  const hasWorkspace = Boolean(snapshot.workspaceRoot);
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
    onOpenWorkspace(path);
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
  const handleCancelRun = async (run: RunSummary): Promise<void> => {
    if (terminalRunStatuses.has(run.status)) return;
    const confirmed = await confirm({
      title: "Mark run as cancelled?",
      description: (
        <>
          Run <code className="rounded bg-muted px-1 py-0.5 text-xs">{run.id}</code> will be marked
          cancelled in the workspace. This does not stop any underlying scheduler job.
        </>
      ),
      confirmLabel: "Mark cancelled",
      destructive: true,
    });
    if (!confirmed) return;
    try {
      await workspaceApi.updateRunStatus(run.projectId, run.experimentId, run.id, "cancelled");
      onRefresh();
    } catch (error) {
      console.error("Failed to mark run cancelled:", error);
      void alert({
        title: "Failed to mark run cancelled",
        description: error instanceof Error ? error.message : String(error),
      });
    }
  };
  const handleDeleteProject = async (projectId: string): Promise<void> => {
    const confirmed = await confirm({
      title: "Delete project?",
      description: (
        <>
          Project <code className="rounded bg-muted px-1 py-0.5 text-xs">{projectId}</code> and its
          experiments will be removed from the workspace.
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
          Experiment <code className="rounded bg-muted px-1 py-0.5 text-xs">{experiment.id}</code>{" "}
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

  const projectTreeActions: ProjectTreeActions = {
    onSelect,
    onCreateExperiment: setCreateExperimentProjectId,
    onCreateRun: setCreateRunExperimentId,
    onDeleteProject: (projectId) => {
      void handleDeleteProject(projectId);
    },
    onDeleteExperiment: (experiment) => {
      void handleDeleteExperiment(experiment);
    },
    onCancelRun: (run) => {
      void handleCancelRun(run);
    },
    onOpenRunView: handleOpenRunView,
    onCopyText: handleCopyText,
    onRefresh,
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
    onRefresh,
  };

  const projectNodes = buildProjectNodes(snapshot, projectTreeActions, searchQuery);
  const workspaceNodes = buildWorkspaceNodes(snapshot, workspaceTreeActions);
  const assetNodes = buildAssetNodes(snapshot, onSelect, handleCopyText, searchQuery);
  const workflowNodes = buildWorkflowNodes(snapshot, onSelect, handleCopyText, searchQuery);
  const agentNodes = buildAgentNodes(snapshot, onSelect, handleCopyText);

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
        emptyTitle={searchQuery ? EMPTY_COPY.projectsFilter.title : EMPTY_COPY.entries.title}
      />
    ),
    workspace: (
      <TreeView
        nodes={workspaceNodes}
        activeId={activeId}
        expandPath={workspaceExpandPath}
        emptyTitle={EMPTY_COPY.workspace.title}
      />
    ),
    runs: (
      <RunsFacetPanel
        facets={runsFacets}
        filters={runsFilters}
        onFiltersChange={handleRunsFiltersChange}
      />
    ),
    asset: (
      <TreeView
        nodes={assetNodes}
        activeId={activeId}
        emptyTitle={EMPTY_COPY.assets.title}
        emptyDescription={EMPTY_COPY.assets.description}
      />
    ),
    workflow: (
      <TreeView nodes={workflowNodes} activeId={activeId} emptyTitle={EMPTY_COPY.entries.title} />
    ),
    agent: (
      <TreeView
        nodes={agentNodes}
        activeId={activeId}
        emptyIcon={<Sparkles className="h-8 w-8" />}
        emptyTitle={EMPTY_COPY.agentSessions.title}
        emptyDescription={EMPTY_COPY.agentSessions.description}
      />
    ),
    settings: (
      <nav className="space-y-0.5 px-1 pb-4 text-xs">
        <div className="rounded-sm bg-muted/30 px-2 py-1.5 font-medium text-foreground">
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
                  <Button
                    variant={isActive ? "secondary" : "ghost"}
                    size="icon"
                    onClick={() => onViewChange(option.id)}
                  >
                    <option.icon className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="right">{option.label}</TooltipContent>
              </Tooltip>
            );
          })}
          <div className="mt-auto">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant={view === "settings" ? "secondary" : "ghost"}
                  size="icon"
                  onClick={() => onViewChange("settings")}
                  aria-label="Settings"
                >
                  <Settings className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="right">Settings</TooltipContent>
            </Tooltip>
          </div>
        </div>
      </TooltipProvider>

      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <div className="space-y-1 px-4 py-3">
          <div className="flex items-center justify-between gap-2">
            <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              {listHeader}
            </p>

            {view === "projects" && (
              <div className="flex items-center gap-1">
                <CreateProjectDialog onProjectCreated={onRefresh} />
              </div>
            )}

            {view === "workspace" && (
              <div className="flex items-center gap-1">
                {!hasWorkspace ? (
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7"
                    onClick={() => {
                      void handleOpenWorkspace();
                    }}
                    aria-label="Open workspace"
                  >
                    <FolderOpen className="h-4 w-4" />
                  </Button>
                ) : (
                  <>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7"
                      onClick={() => {
                        void handleCreateFile();
                      }}
                      aria-label="New file"
                    >
                      <FilePlus className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7"
                      onClick={() => {
                        void handleCreateDirectory();
                      }}
                      aria-label="New folder"
                    >
                      <FolderPlus className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7"
                      onClick={onRefresh}
                      aria-label="Refresh workspace"
                    >
                      <RefreshCw className="h-4 w-4" />
                    </Button>
                  </>
                )}
              </div>
            )}

            {view === "agent" && (
              <div className="flex items-center gap-1">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7"
                  onClick={() => onSelect({ objectType: "agent", objectId: "settings" })}
                  aria-label="Agent settings"
                  title="Skills, tools, MCP"
                >
                  <Settings className="h-4 w-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7"
                  onClick={() => onSelect({ objectType: "agent", objectId: "new" })}
                  aria-label="New goal"
                >
                  <Plus className="h-4 w-4" />
                </Button>
              </div>
            )}
          </div>
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
          onRunCreated={onRefresh}
        />
      )}
      {promptDialog}
      {confirmDialog}
      {alertDialog}
    </div>
  );
};
