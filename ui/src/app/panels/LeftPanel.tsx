import {
  Archive,
  Blocks,
  Bot,
  Clock,
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
  Workflow,
} from "lucide-react";
import type { ComponentType, SVGProps } from "react";
import { useEffect, useMemo, useRef } from "react";
import { CreateProjectDialog } from "@/app/components/CreateProjectDialog";
import { EMPTY_COPY, StatusBadge } from "@/app/components/entity";
import type { TreeNode } from "@/app/panels/TreeView";
import { TreeView } from "@/app/panels/TreeView";
import type {
  FileKind,
  LeftPanelView,
  Selection,
  SemanticStatus,
  WorkspaceSnapshot,
  WorkspaceTreeNode,
} from "@/app/types";
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
  { id: "asset", label: "Asset", icon: Archive },
  { id: "workflow", label: "Workflow", icon: Workflow },
  { id: "agent", label: "Agent", icon: Bot },
];

const listHeaderByView: Record<LeftPanelView, string> = {
  projects: "Projects",
  workspace: "Workspace",
  asset: "Assets",
  workflow: "Workflows",
  agent: "Agent Sessions",
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

const detectFileKind = (path: string): FileKind => {
  const parts = path.split(".");
  const extension = parts.length > 1 ? `.${parts[parts.length - 1].toLowerCase()}` : "";
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

const buildProjectNodes = (
  snapshot: WorkspaceSnapshot,
  onSelect: (selection: Selection) => void,
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
    meta: project.experiments.length,
    onSelect: () => onSelect({ objectType: "project", objectId: project.id }),
    children: project.experiments.map((experiment) => ({
      id: experiment.id,
      label: experiment.name,
      icon: FlaskConical,
      iconClassName: "text-purple-500",
      right: <StatusBadge status={experiment.status} size="sm" />,
      onSelect: () => onSelect({ objectType: "experiment", objectId: experiment.id }),
      emptyChildLabel: EMPTY_COPY.runs.title,
      children: experiment.runs.map((run) => ({
        id: run.id,
        label: run.name || run.id,
        icon: PlayCircle,
        iconClassName: runIconClass(run.status),
        meta: run.id.substring(0, 8),
        onSelect: () => onSelect({ objectType: "run", objectId: run.id }),
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
  onSelect: (selection: Selection) => void,
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
          onSelect({
            objectType: "workspace-file",
            objectId: node.path,
            filePath: node.path,
            fileKind: detectFileKind(node.path),
          });
          return;
        }
        if (semantic) {
          onSelect({ objectType: semantic.type, objectId: semantic.id });
        }
      },
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
  searchQuery: string,
): TreeNode[] => {
  return filterBySearch(snapshot.assets, searchQuery).map((asset) => ({
    id: asset.id,
    label: asset.name,
    icon: Archive,
    iconClassName: "text-amber-500",
    right: <StatusBadge status={asset.status} size="sm" />,
    onSelect: () => onSelect({ objectType: "asset", objectId: asset.id }),
  }));
};

const buildWorkflowNodes = (
  snapshot: WorkspaceSnapshot,
  onSelect: (selection: Selection) => void,
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
  }));
};

const buildAgentNodes = (
  snapshot: WorkspaceSnapshot,
  onSelect: (selection: Selection) => void,
): TreeNode[] => {
  return snapshot.agentSessions.map((session) => ({
    id: session.id,
    label: session.goalDescription,
    icon: Bot,
    iconClassName: "text-violet-500",
    right: <StatusBadge status={session.status} size="sm" />,
    meta: (
      <span className="flex items-center gap-0.5">
        <Clock className="h-2.5 w-2.5" />
        {new Date(session.createdAt).toLocaleDateString()}
      </span>
    ),
    onSelect: () => onSelect({ objectType: "agent", objectId: session.id }),
  }));
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

  const handleOpenWorkspace = (): void => {
    const path = window.prompt("Workspace path");
    if (!path) return;
    onOpenWorkspace(path);
  };
  const handleCreateFile = (): void => {
    const path = window.prompt("New file path (relative to workspace)");
    if (!path) return;
    onCreateFile(path);
  };
  const handleCreateDirectory = (): void => {
    const path = window.prompt("New folder path (relative to workspace)");
    if (!path) return;
    onCreateDirectory(path);
  };

  const projectNodes = useMemo(
    () => buildProjectNodes(snapshot, onSelect, searchQuery),
    [snapshot, onSelect, searchQuery],
  );
  const workspaceNodes = useMemo(
    () => buildWorkspaceNodes(snapshot, onSelect),
    [snapshot, onSelect],
  );
  const assetNodes = useMemo(
    () => buildAssetNodes(snapshot, onSelect, searchQuery),
    [snapshot, onSelect, searchQuery],
  );
  const workflowNodes = useMemo(
    () => buildWorkflowNodes(snapshot, onSelect, searchQuery),
    [snapshot, onSelect, searchQuery],
  );
  const agentNodes = useMemo(() => buildAgentNodes(snapshot, onSelect), [snapshot, onSelect]);

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
  };

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
                <Button variant="ghost" size="icon">
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
                    onClick={handleOpenWorkspace}
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
                      onClick={handleCreateFile}
                      aria-label="New file"
                    >
                      <FilePlus className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7"
                      onClick={handleCreateDirectory}
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
    </div>
  );
};
