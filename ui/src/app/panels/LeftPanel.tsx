import {
  Archive,
  Blocks,
  ChevronRight,
  FileText,
  FilePlus,
  FlaskConical,
  Folder,
  FolderOpen,
  FolderPlus,
  FolderTree,
  PlayCircle,
  RefreshCw,
  Workflow,
  Settings,
} from "lucide-react";
import { useEffect, useState, useRef, useMemo } from "react";

import type { ComponentType, SVGProps } from "react";
import type {
  AssetSummary,
  FileKind,
  LeftPanelView,
  ProjectSummary,
  Selection,
  SemanticStatus,
  WorkspaceSnapshot,
  WorkspaceTreeNode,
  WorkflowSummary,
} from "@/app/types";
import { CreateProjectDialog } from "@/app/components/CreateProjectDialog";
import { Badge } from "@/components/ui/badge";
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
  { id: "workspace", label: "Workspace", icon: FolderTree },
  { id: "project", label: "Project", icon: Blocks },
  { id: "experiment", label: "Experiment", icon: FlaskConical },
  { id: "run", label: "Run", icon: PlayCircle }, // Fixed duplicate
  { id: "asset", label: "Asset", icon: Archive },
  { id: "workflow", label: "Workflow", icon: Workflow },
];

const statusStyles: Record<SemanticStatus, string> = {
  active: "bg-emerald-100 text-emerald-900",
  archived: "bg-slate-200 text-slate-900",
  draft: "bg-amber-100 text-amber-900",
  pending: "bg-slate-100 text-slate-900",
  running: "bg-blue-100 text-blue-900",
  succeeded: "bg-emerald-100 text-emerald-900",
  failed: "bg-rose-100 text-rose-900",
  cancelled: "bg-slate-200 text-slate-900",
  skipped: "bg-amber-100 text-amber-900",
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

const buildListHeader = (view: LeftPanelView): string => {
  const labelByView: Record<LeftPanelView, string> = {
    workspace: "Workspace",
    project: "Projects",
    experiment: "Experiments",
    run: "Runs",
    asset: "Assets",
    workflow: "Workflows",
  };

  return labelByView[view];
};

const WorkspaceTree = ({
  root,
  onSelect,
  activePath,
  snapshot,
}: {
  root: WorkspaceTreeNode | null;
  onSelect: (selection: Selection) => void;
  activePath?: string;
  snapshot: WorkspaceSnapshot;
}): JSX.Element => {
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (root) {
      setExpandedIds(new Set([root.id]));
    }
  }, [root]);

  if (!root) {
    return (
      <div className="space-y-2 text-sm text-muted-foreground">
        <p>No workspace files loaded.</p>
      </div>
    );
  }



  const isExpanded = (nodeId: string): boolean => {
    return expandedIds.has(nodeId);
  };

  const toggleExpanded = (nodeId: string): void => {
    setExpandedIds(prev => {
      const next = new Set(prev);
      if (next.has(nodeId)) {
        next.delete(nodeId);
      } else {
        next.add(nodeId);
      }
      return next;
    });
  };

  // Helper to detect if a node corresponds to a semantic object
  const getSemanticInfo = (path: string) => {
    // Check Projects
    const project = snapshot.projects.find(p => path.endsWith(`projects/${p.id}`));
    if (project) return { type: 'project' as const, id: project.id, icon: Blocks };

    // Check Experiments
    const experiment = snapshot.experiments.find(e => path.endsWith(`experiments/${e.id}`));
    if (experiment) return { type: 'experiment' as const, id: experiment.id, icon: FlaskConical };
    
    // Check Runs
    // Path: .../runs/<rid>
    const run = snapshot.runs.find(r => path.endsWith(`runs/${r.id}`));
    if (run) return { type: 'run' as const, id: run.id, icon: PlayCircle, status: run.status };

    // Check Assets (Workspace or Project level)
    const pathParts = path.split('/');
    const folderName = pathParts[pathParts.length - 1];
    const parentName = pathParts.length > 1 ? pathParts[pathParts.length - 2] : null;
    
    if (parentName === 'assets') {
         const asset = snapshot.assets.find(a => a.id === folderName);
         if (asset) return { type: 'asset' as const, id: asset.id, icon: Archive };
    }

    return null;
  };

  const renderNode = (node: WorkspaceTreeNode, depth: number): JSX.Element => {
    const isFile = node.kind === "file";
    const expanded = isExpanded(node.id);
    const hasChildren = node.children.length > 0;
    const isActive = activePath === node.path;
    
    // Check for semantic meaning
    const semantic = !isFile ? getSemanticInfo(node.path) : null;
    const NodeIcon = semantic ? semantic.icon : (isFile ? FileText : (expanded ? FolderOpen : Folder));
    
    // Dynamic indentation: 16px per level (standard pl-4 equivalent)
    // depth 0 = 0px, depth 1 = 16px, etc.
    const paddingLeft = `${Math.max(0, depth) * 16}px`;

    return (
      <div key={node.id} className="space-y-1">
        <div className="flex items-center gap-1" style={{ paddingLeft }}>
          {!isFile && (
            <button
              type="button"
              aria-label={expanded ? "Collapse folder" : "Expand folder"}
              className="flex h-6 w-6 items-center justify-center rounded-sm text-muted-foreground transition-colors hover:bg-muted/40 hover:text-foreground flex-none"
              onClick={(e) => {
                 e.stopPropagation();
                 toggleExpanded(node.id);
              }}
            >
              <ChevronRight
                className={`h-3.5 w-3.5 transition-transform ${
                  expanded ? "rotate-90" : "rotate-0"
                }`}
              />
            </button>
          )}
          {isFile && <span className="h-6 w-6 flex-none" />}
          
          <Button
            variant={isActive ? "secondary" : "ghost"}
            className="h-7 flex-1 justify-start text-xs overflow-hidden group" 
            onClick={() => {
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
                  // Navigate to dashboard without expanding
                  onSelect({
                      objectType: semantic.type,
                      objectId: semantic.id
                  });
              } else {
                  // Default folder behavior: toggle expand
                  toggleExpanded(node.id);
              }
            }}
            title={node.name}
          >
            <NodeIcon className={`mr-2 h-3.5 w-3.5 flex-none ${
                semantic?.type === 'project' ? 'text-blue-500' : 
                semantic?.type === 'experiment' ? 'text-purple-500' :
                semantic?.type === 'run' ? (
                     semantic.status === 'succeeded' ? 'text-green-500' :
                     semantic.status === 'failed' ? 'text-red-500' :
                     semantic.status === 'running' ? 'text-blue-500' : 'text-muted-foreground'
                ) :
                semantic?.type === 'asset' ? 'text-amber-500' : ''
            }`} />
            <span className="truncate flex-1 text-left">{node.name}</span>
            
            {semantic && (
                 <span className="text-[10px] text-muted-foreground uppercase tracking-tighter ml-auto opacity-50 group-hover:opacity-100 transition-opacity">
                    {semantic.type.substring(0,3)}
                 </span>
            )}
          </Button>
        </div>
        {!isFile && expanded && hasChildren && (
          <div className="space-y-1">
            {node.children.map(child => renderNode(child, depth + 1))}
          </div>
        )}
        {!isFile && expanded && !hasChildren && (
          <div className="text-xs text-muted-foreground pl-8" style={{ paddingLeft: `${(depth + 1) * 16}px` }}>
            Empty folder
          </div>
        )}
      </div>
    );
  };

  return <div className="space-y-1">{renderNode(root, 0)}</div>;
};

const SemanticList = <T extends { id: string; name: string; status: SemanticStatus; summary: string }>(
  {
    items,
    onSelect,
    objectType,
    activeId,
  }: {
    items: T[];
    onSelect: (selection: Selection) => void;
    objectType: "project" | "experiment" | "run" | "asset" | "workflow";
    activeId?: string;
  },
): JSX.Element => {
  if (items.length === 0) {
    return <p className="text-xs text-muted-foreground">No entries available.</p>;
  }

  return (
    <div className="space-y-3">
      {items.map(item => (
        <button
          key={item.id}
          type="button"
          className={`w-full rounded-md border text-left transition-colors ${
              activeId === item.id 
              ? "border-primary/50 bg-primary/5" 
              : "border-border/60 bg-background hover:bg-muted/40"
          } px-3 py-2 overflow-hidden`}
          onClick={() => {
            if (objectType === "workflow") {
              onSelect({ objectType: "workflow", objectId: item.id, workflowId: item.id });
              return;
            }
            onSelect({ objectType, objectId: item.id });
          }}
        >
          <div className="flex items-start justify-between gap-2">
            <div className="min-w-0 flex-1">
              <p className="text-sm font-semibold text-foreground truncate" title={item.name}>{item.name}</p>
              <p className="text-xs text-muted-foreground truncate" title={item.summary}>{item.summary}</p>
            </div>
            <Badge className={`mt-0.5 flex-none ${statusStyles[item.status]}`}>{item.status}</Badge>
          </div>
        </button>
      ))}
    </div>
  );
};

const RunTree = ({
  snapshot,
  onSelect,
  activeId,
  searchQuery,
}: {
  snapshot: WorkspaceSnapshot;
  onSelect: (selection: Selection) => void;
  activeId?: string;
  searchQuery?: string;
}): JSX.Element => {
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());

  // Build Hierarchy
  const hierarchy = useMemo(() => {
    const runs = snapshot.runs;
    const lowerQuery = searchQuery?.toLowerCase() || "";
    
    return snapshot.projects.map(p => ({
        ...p,
        experiments: snapshot.experiments.filter(e => e.projectId === p.id).map(e => ({
            ...e,
            runs: runs.filter(r => r.experimentId === e.id).filter(r => 
                !lowerQuery || r.name.toLowerCase().includes(lowerQuery) || r.id.includes(lowerQuery)
            )
        })).filter(e => e.runs.length > 0)
    })).filter(p => p.experiments.length > 0);
  }, [snapshot, searchQuery]);

  // Auto-expand if searching
  useEffect(() => {
    if (searchQuery) {
        const allIds = new Set<string>();
        hierarchy.forEach(p => {
             allIds.add(p.id);
             p.experiments.forEach(e => allIds.add(e.id));
        });
        setExpandedIds(allIds);
    }
  }, [hierarchy, searchQuery]);

  const toggleExpanded = (id: string) => {
    setExpandedIds(prev => {
        const next = new Set(prev);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        return next;
    });
  };

  if (hierarchy.length === 0) {
      if (searchQuery) return <p className="text-xs text-muted-foreground px-4">No runs match "{searchQuery}".</p>;
      return <p className="text-xs text-muted-foreground">No runs found.</p>;
  }

  return (
    <div className="space-y-1">
        {hierarchy.map(project => (
            <div key={project.id} className="space-y-1">
                {/* Project Node */}
                <button
                    type="button"
                    className="flex w-full items-center gap-1 rounded-sm px-2 py-1 text-sm font-medium hover:bg-muted/40"
                    onClick={() => toggleExpanded(project.id)}
                >
                    <ChevronRight className={`h-3.5 w-3.5 transition-transform ${expandedIds.has(project.id) ? "rotate-90" : ""}`} />
                    <Folder className="mr-2 h-3.5 w-3.5 text-blue-500" />
                    <span className="truncate flex-1 text-left min-w-0" title={project.name}>{project.name}</span>
                    <span className="text-xs text-muted-foreground">{project.experiments.length}</span>
                </button>

                {expandedIds.has(project.id) && (
                    <div className="pl-4 space-y-1">
                        {project.experiments.map(exp => (
                            <div key={exp.id} className="space-y-1">
                                {/* Experiment Node */}
                                <button
                                    type="button"
                                    className="flex w-full items-center gap-1 rounded-sm px-2 py-1 text-sm hover:bg-muted/40"
                                    onClick={() => toggleExpanded(exp.id)}
                                >
                                     <ChevronRight className={`h-3.5 w-3.5 transition-transform ${expandedIds.has(exp.id) ? "rotate-90" : ""}`} />
                                     <FlaskConical className="mr-2 h-3.5 w-3.5 text-purple-500" />
                                     <span className="truncate flex-1 text-left min-w-0" title={exp.name}>{exp.name}</span>
                                     <span className="text-xs text-muted-foreground">{exp.runs.length}</span>
                                </button>

                                {expandedIds.has(exp.id) && (
                                    <div className="pl-6 space-y-1">
                                        {exp.runs.map(run => (
                                            <button
                                                key={run.id}
                                                type="button"
                                                className={`flex w-full items-center gap-2 rounded-sm px-2 py-1 text-sm transition-colors overflow-hidden ${
                                                    activeId === run.id ? "bg-accent text-accent-foreground" : "hover:bg-muted/40"
                                                }`}
                                                onClick={() => onSelect({ objectType: "run", objectId: run.id })}
                                            >
                                                <div className={`h-2 w-2 rounded-full flex-none ${
                                                    run.status === 'succeeded' ? 'bg-green-500' :
                                                    run.status === 'failed' ? 'bg-red-500' :
                                                    run.status === 'running' ? 'bg-blue-500' : 'bg-gray-400'
                                                }`} />
                                                <span className="truncate flex-1 text-left min-w-0" title={run.name}>{run.name || run.id}</span>
                                                <span className="text-[10px] text-muted-foreground font-mono flex-none">{run.id.substring(0,6)}</span>
                                            </button>
                                        ))}
                                        {exp.runs.length === 0 && (
                                            <p className="pl-6 text-xs text-muted-foreground py-1">No runs</p>
                                        )}
                                    </div>
                                )}
                            </div>
                        ))}
                         {project.experiments.length === 0 && (
                             <p className="pl-6 text-xs text-muted-foreground py-1">No experiments</p>
                        )}
                    </div>
                )}
            </div>
        ))}
    </div>
  );
};

const ExperimentTree = ({
  snapshot,
  onSelect,
  activeId,
  searchQuery,
}: {
  snapshot: WorkspaceSnapshot;
  onSelect: (selection: Selection) => void;
  activeId?: string;
  searchQuery?: string;
}): JSX.Element => {
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());

  const hierarchy = useMemo(() => {
    const lowerQuery = searchQuery?.toLowerCase() || "";
    
    const projects = snapshot.projects.map(p => ({
        ...p,
        experiments: snapshot.experiments.filter(e => 
            e.projectId === p.id && 
            (!lowerQuery || e.name.toLowerCase().includes(lowerQuery) || e.summary?.toLowerCase().includes(lowerQuery))
        )
    })).filter(p => p.experiments.length > 0);
    
    return projects;
  }, [snapshot, searchQuery]);

  // Auto-expand if searching
  useEffect(() => {
    if (searchQuery) {
        const allIds = new Set<string>();
        hierarchy.forEach(p => allIds.add(p.id));
        setExpandedIds(allIds);
    }
  }, [hierarchy, searchQuery]);

  const toggleExpanded = (id: string) => {
    setExpandedIds(prev => {
        const next = new Set(prev);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        return next;
    });
  };

  if (hierarchy.length === 0) {
      if (searchQuery) return <p className="text-xs text-muted-foreground px-4">No experiments match "{searchQuery}".</p>;
      return <p className="text-xs text-muted-foreground">No experiments found.</p>;
  }

  return (
    <div className="space-y-1">
        {hierarchy.map(project => (
            <div key={project.id} className="space-y-1">
                {/* Project Node */}
                <button
                    type="button"
                    className="flex w-full items-center gap-1 rounded-sm px-2 py-1 text-sm font-medium hover:bg-muted/40"
                    onClick={() => toggleExpanded(project.id)}
                >
                    <ChevronRight className={`h-3.5 w-3.5 transition-transform ${expandedIds.has(project.id) ? "rotate-90" : ""}`} />
                    <Folder className="mr-2 h-3.5 w-3.5 text-blue-500" />
                    <span className="truncate flex-1 text-left min-w-0" title={project.name}>{project.name}</span>
                    <span className="text-xs text-muted-foreground">{project.experiments.length}</span>
                </button>

                {expandedIds.has(project.id) && (
                    <div className="pl-4 space-y-1">
                        {project.experiments.map(exp => (
                            <button
                                key={exp.id}
                                type="button"
                                className={`flex w-full items-center gap-2 rounded-sm px-2 py-1 text-sm transition-colors overflow-hidden ${
                                    activeId === exp.id ? "bg-accent text-accent-foreground" : "hover:bg-muted/40"
                                }`}
                                onClick={() => onSelect({ objectType: "experiment", objectId: exp.id })}
                            >
                                <FlaskConical className="h-3.5 w-3.5 text-purple-500 flex-none" />
                                <span className="truncate flex-1 text-left min-w-0" title={exp.name}>{exp.name}</span>
                                <Badge variant="secondary" className={`text-[10px] h-4 px-1 flex-none ${statusStyles[exp.status]}`}>
                                    {exp.status}
                                </Badge>
                            </button>
                        ))}
                        {project.experiments.length === 0 && (
                             <p className="pl-6 text-xs text-muted-foreground py-1">No experiments</p>
                        )}
                    </div>
                )}
            </div>
        ))}
    </div>
  );
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
  const listHeader = buildListHeader(view);
  const hasWorkspace = Boolean(snapshot.workspaceRoot);

  // Track previous selection to only sync on change
  const prevSelectionRef = useRef(selection);

  // Sync view only when selection VALUE changes
  useEffect(() => {
    const prev = prevSelectionRef.current;
    const isSameSelection = 
        (prev === selection) || 
        (prev?.objectId === selection?.objectId && prev?.objectType === selection?.objectType);

    if (isSameSelection) return;
    
    prevSelectionRef.current = selection;
  }, [selection]);

  const activeId = selection ? selection.objectId : undefined;
  const activePath = selection?.objectType === 'workspace-file' ? selection.filePath : undefined;

  const handleOpenWorkspace = (): void => {
    const path = window.prompt("Workspace path");
    if (!path) {
      return;
    }
    onOpenWorkspace(path);
  };

  const handleCreateFile = (): void => {
    const path = window.prompt("New file path (relative to workspace)");
    if (!path) {
      return;
    }
    onCreateFile(path);
  };

  const handleCreateDirectory = (): void => {
    const path = window.prompt("New folder path (relative to workspace)");
    if (!path) {
      return;
    }
    onCreateDirectory(path);
  };

  const filterItems = <T extends { name: string; summary?: string }>(items: T[]) => {
    if (!searchQuery) return items;
    const lowerQuery = searchQuery.toLowerCase();
    return items.filter(item => 
      item.name.toLowerCase().includes(lowerQuery) || 
      item.summary?.toLowerCase().includes(lowerQuery)
    );
  };

  const listContent: Record<LeftPanelView, JSX.Element> = {
    workspace: <WorkspaceTree root={snapshot.workspaceRoot} onSelect={onSelect} activePath={activePath} snapshot={snapshot} />,
    project: (
      <div className="space-y-4">
        <div className="flex justify-end px-1">
          <CreateProjectDialog onProjectCreated={onRefresh} />
        </div>
        <SemanticList<ProjectSummary>
          items={filterItems(snapshot.projects)}
          onSelect={onSelect}
          objectType="project"
          activeId={activeId}
        />
      </div>
    ),
    experiment: (
      <div className="space-y-6">
         <ExperimentTree snapshot={snapshot} onSelect={onSelect} activeId={activeId} searchQuery={searchQuery} />
      </div>
    ),
    run: (
      <RunTree snapshot={snapshot} onSelect={onSelect} activeId={activeId} searchQuery={searchQuery} />
    ),
    asset: (
      <SemanticList<AssetSummary>
        items={filterItems(snapshot.assets)}
        onSelect={onSelect}
        objectType="asset"
      />
    ),
    workflow: (
      <SemanticList<WorkflowSummary>
        items={filterItems(snapshot.workflows)}
        onSelect={onSelect}
        objectType="workflow"
        activeId={activeId}
      />
    ),
  };

  return (
    <div className="flex h-full">
      <TooltipProvider>
        <div className="flex w-14 flex-col items-center gap-2 border-r border-border bg-muted/20 py-4">
          {viewOptions.map(option => {
            const isActive = view === option.id;
            return (
              <Tooltip key={option.id}>
                <TooltipTrigger asChild>
                  <Button
                    variant={isActive ? "secondary" : "ghost"}
                    size="icon"
                    onClick={() => {
                      onViewChange(option.id);
                    }}
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

      <div className="flex flex-1 flex-col min-w-0 overflow-hidden">
        <div className="space-y-1 px-4 py-3">
          <div className="flex items-center justify-between gap-2">
            <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              {listHeader}
            </p>
            {view === "workspace" && (
              <div className="flex items-center gap-1">
                {!hasWorkspace ? (
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-7"
                    onClick={handleOpenWorkspace}
                  >
                    <FolderOpen className="mr-1 h-3.5 w-3.5" />
                    Open
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
          </div>
          <Separator />
        </div>
        <ScrollArea className="flex-1 px-4 pb-4">{listContent[view]}</ScrollArea>
      </div>
    </div>
  );
};
