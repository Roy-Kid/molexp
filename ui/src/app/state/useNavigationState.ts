import { useCallback, useMemo } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import type {
  BreadcrumbItem,
  LeftPanelView,
  Selection,
  WorkspaceSnapshot,
} from "@/app/types";

const sectionRootByView: Record<LeftPanelView, string> = {
  projects: "/projects",
  workspace: "/workspace",
  workflow: "/workflows",
  asset: "/assets",
  agent: "/agents",
};

const buildWorkspaceFileSelection = (searchParams: URLSearchParams): Selection | null => {
  const filePath = searchParams.get("file");
  const fileKind = searchParams.get("fileKind");

  if (!filePath || !fileKind) {
    return null;
  }

  return {
    objectType: "workspace-file",
    objectId: filePath,
    filePath,
    fileKind:
      fileKind === "yaml" ||
      fileKind === "json" ||
      fileKind === "python" ||
      fileKind === "markdown" ||
      fileKind === "text" ||
      fileKind === "image"
        ? fileKind
        : "unknown",
  };
};

const getLeftPanelViewFromPath = (pathname: string): LeftPanelView => {
  if (pathname.startsWith("/workspace")) {
    return "workspace";
  }
  if (pathname.startsWith("/workflows")) {
    return "workflow";
  }
  if (pathname.startsWith("/assets")) {
    return "asset";
  }
  if (pathname.startsWith("/agents")) {
    return "agent";
  }
  return "projects";
};

const buildSelectionFromLocation = (
  pathname: string,
  searchParams: URLSearchParams,
): Selection | null => {
  const projectRunMatch = pathname.match(
    /^\/projects\/([^/]+)\/experiments\/([^/]+)\/runs\/([^/]+)$/,
  );
  if (projectRunMatch) {
    return {
      objectType: "run",
      objectId: decodeURIComponent(projectRunMatch[3]),
    };
  }

  const experimentMatch = pathname.match(/^\/projects\/([^/]+)\/experiments\/([^/]+)$/);
  if (experimentMatch) {
    return {
      objectType: "experiment",
      objectId: decodeURIComponent(experimentMatch[2]),
    };
  }

  const projectMatch = pathname.match(/^\/projects\/([^/]+)$/);
  if (projectMatch) {
    return {
      objectType: "project",
      objectId: decodeURIComponent(projectMatch[1]),
    };
  }

  const workflowMatch = pathname.match(/^\/workflows\/([^/]+)$/);
  if (workflowMatch) {
    return {
      objectType: "workflow",
      objectId: decodeURIComponent(workflowMatch[1]),
      workflowId: decodeURIComponent(workflowMatch[1]),
    };
  }

  const assetMatch = pathname.match(/^\/assets\/([^/]+)$/);
  if (assetMatch) {
    return {
      objectType: "asset",
      objectId: decodeURIComponent(assetMatch[1]),
    };
  }

  if (pathname === "/agents/new") {
    return {
      objectType: "agent",
      objectId: "new",
    };
  }

  const agentMatch = pathname.match(/^\/agents\/([^/]+)$/);
  if (agentMatch) {
    return {
      objectType: "agent",
      objectId: decodeURIComponent(agentMatch[1]),
    };
  }

  if (pathname.startsWith("/workspace")) {
    return buildWorkspaceFileSelection(searchParams);
  }

  return null;
};

const getSelectionPath = (selection: Selection | null, snapshot: WorkspaceSnapshot): string => {
  if (!selection) {
    return "/projects";
  }

  switch (selection.objectType) {
    case "project":
      return `/projects/${encodeURIComponent(selection.objectId)}`;
    case "experiment": {
      const experiment = snapshot.experiments.find((item) => item.id === selection.objectId);
      if (!experiment) {
        return "/projects";
      }
      return `/projects/${encodeURIComponent(experiment.projectId)}/experiments/${encodeURIComponent(experiment.id)}`;
    }
    case "run": {
      const run = snapshot.runs.find((item) => item.id === selection.objectId);
      if (!run) {
        return "/projects";
      }
      return `/projects/${encodeURIComponent(run.projectId)}/experiments/${encodeURIComponent(run.experimentId)}/runs/${encodeURIComponent(run.id)}`;
    }
    case "workflow":
      return `/workflows/${encodeURIComponent(selection.workflowId)}`;
    case "asset":
      return `/assets/${encodeURIComponent(selection.objectId)}`;
    case "agent":
      return selection.objectId === "new"
        ? "/agents/new"
        : `/agents/${encodeURIComponent(selection.objectId)}`;
    case "workspace-file": {
      const params = new URLSearchParams({
        file: selection.filePath,
        fileKind: selection.fileKind,
      });
      return `/workspace?${params.toString()}`;
    }
  }
};

const buildBreadcrumbs = (
  selection: Selection | null,
  snapshot: WorkspaceSnapshot,
  leftPanelView: LeftPanelView,
): BreadcrumbItem[] => {
  if (!selection) {
    if (leftPanelView === "projects") {
      return [{ label: "Projects" }];
    }
    if (leftPanelView === "workflow") {
      return [{ label: "Workflows" }];
    }
    if (leftPanelView === "asset") {
      return [{ label: "Assets" }];
    }
    if (leftPanelView === "agent") {
      return [{ label: "Agents" }];
    }
    return [{ label: "Workspace" }];
  }

  switch (selection.objectType) {
    case "project": {
      const project = snapshot.projects.find((item) => item.id === selection.objectId);
      return [
        { label: "Projects", to: "/projects" },
        { label: project?.name ?? selection.objectId },
      ];
    }
    case "experiment": {
      const experiment = snapshot.experiments.find((item) => item.id === selection.objectId);
      const project = experiment
        ? snapshot.projects.find((item) => item.id === experiment.projectId)
        : null;
      return [
        { label: "Projects", to: "/projects" },
        ...(project
          ? [
              {
                label: project.name,
                to: `/projects/${encodeURIComponent(project.id)}`,
              },
            ]
          : []),
        { label: experiment?.name ?? selection.objectId },
      ];
    }
    case "run": {
      const run = snapshot.runs.find((item) => item.id === selection.objectId);
      const experiment = run
        ? snapshot.experiments.find((item) => item.id === run.experimentId)
        : null;
      const project = run ? snapshot.projects.find((item) => item.id === run.projectId) : null;
      return [
        { label: "Projects", to: "/projects" },
        ...(project
          ? [
              {
                label: project.name,
                to: `/projects/${encodeURIComponent(project.id)}`,
              },
            ]
          : []),
        ...(project && experiment
          ? [
              {
                label: experiment.name,
                to: `/projects/${encodeURIComponent(project.id)}/experiments/${encodeURIComponent(experiment.id)}`,
              },
            ]
          : []),
        { label: run?.name ?? selection.objectId },
      ];
    }
    case "workflow": {
      const workflow = snapshot.workflows.find((item) => item.id === selection.workflowId);
      return [
        { label: "Workflows", to: "/workflows" },
        { label: workflow?.name ?? selection.workflowId },
      ];
    }
    case "asset": {
      const asset = snapshot.assets.find((item) => item.id === selection.objectId);
      return [{ label: "Assets", to: "/assets" }, { label: asset?.name ?? selection.objectId }];
    }
    case "agent": {
      if (selection.objectId === "new") {
        return [{ label: "Agents", to: "/agents" }, { label: "New Goal" }];
      }

      const session = snapshot.agentSessions.find((item) => item.id === selection.objectId);
      return [
        { label: "Agents", to: "/agents" },
        { label: session?.goalDescription ?? selection.objectId },
      ];
    }
    case "workspace-file":
      return [
        { label: "Workspace", to: "/workspace" },
        { label: selection.filePath.split("/").pop() ?? selection.filePath },
      ];
  }
};

const buildContextMeta = (
  selection: Selection | null,
  snapshot: WorkspaceSnapshot,
  leftPanelView: LeftPanelView,
): { title: string; subtitle: string; statusLabel?: string } => {
  if (!selection) {
    switch (leftPanelView) {
      case "projects":
        return {
          title: "Projects",
          subtitle: "Browse projects, experiments, and runs from a single hierarchy.",
        };
      case "workspace":
        return {
          title: "Workspace",
          subtitle: "Explore source files and workspace artifacts.",
        };
      case "workflow":
        return {
          title: "Workflows",
          subtitle: "Inspect workflow definitions and graph structure.",
        };
      case "asset":
        return {
          title: "Assets",
          subtitle: "Review generated and imported project assets.",
        };
      case "agent":
        return {
          title: "Agents",
          subtitle: "Manage agent sessions and goals.",
        };
    }
  }

  switch (selection.objectType) {
    case "project": {
      const project = snapshot.projects.find((item) => item.id === selection.objectId);
      return {
        title: project?.name ?? selection.objectId,
        subtitle: project?.summary || "Project overview",
        statusLabel: project?.status,
      };
    }
    case "experiment": {
      const experiment = snapshot.experiments.find((item) => item.id === selection.objectId);
      return {
        title: experiment?.name ?? selection.objectId,
        subtitle: experiment?.workflowFile || experiment?.summary || "Experiment overview",
        statusLabel: experiment?.status,
      };
    }
    case "run": {
      const run = snapshot.runs.find((item) => item.id === selection.objectId);
      return {
        title: run?.name ?? selection.objectId,
        subtitle: run?.summary || "Run overview",
        statusLabel: run?.status,
      };
    }
    case "workflow": {
      const workflow = snapshot.workflows.find((item) => item.id === selection.workflowId);
      return {
        title: workflow?.name ?? selection.workflowId,
        subtitle: workflow?.summary || "Workflow overview",
        statusLabel: workflow?.status,
      };
    }
    case "asset": {
      const asset = snapshot.assets.find((item) => item.id === selection.objectId);
      return {
        title: asset?.name ?? selection.objectId,
        subtitle: asset?.summary || "Asset overview",
        statusLabel: asset?.status,
      };
    }
    case "agent": {
      const session = snapshot.agentSessions.find((item) => item.id === selection.objectId);
      return {
        title: selection.objectId === "new" ? "New Goal" : session?.goalDescription ?? selection.objectId,
        subtitle:
          selection.objectId === "new"
            ? "Create a new agent session."
            : `${session?.eventCount ?? 0} events`,
        statusLabel: session?.status,
      };
    }
    case "workspace-file":
      return {
        title: selection.filePath.split("/").pop() ?? selection.filePath,
        subtitle: selection.filePath,
      };
  }
};

export interface NavigationState {
  breadcrumbs: BreadcrumbItem[];
  canNavigateUp: boolean;
  contextStatusLabel?: string;
  contextSubtitle: string;
  contextTitle: string;
  leftPanelView: LeftPanelView;
  selection: Selection | null;
  navigateUp: () => void;
  setLeftPanelView: (view: LeftPanelView) => void;
  setSelection: (selection: Selection | null) => void;
}

export const useNavigationState = (snapshot: WorkspaceSnapshot): NavigationState => {
  const location = useLocation();
  const navigate = useNavigate();

  const searchParams = useMemo(
    () => new URLSearchParams(location.search),
    [location.search],
  );

  const leftPanelView = useMemo(
    () => getLeftPanelViewFromPath(location.pathname),
    [location.pathname],
  );

  const selection = useMemo(
    () => buildSelectionFromLocation(location.pathname, searchParams),
    [location.pathname, searchParams],
  );

  const breadcrumbs = useMemo(
    () => buildBreadcrumbs(selection, snapshot, leftPanelView),
    [selection, snapshot, leftPanelView],
  );

  const contextMeta = useMemo(
    () => buildContextMeta(selection, snapshot, leftPanelView),
    [selection, snapshot, leftPanelView],
  );

  const setSelection = useCallback(
    (nextSelection: Selection | null): void => {
      navigate(getSelectionPath(nextSelection, snapshot));
    },
    [navigate, snapshot],
  );

  const setLeftPanelView = useCallback(
    (view: LeftPanelView): void => {
      if (selection) {
        if (view === "projects" && ["project", "experiment", "run"].includes(selection.objectType)) {
          navigate(getSelectionPath(selection, snapshot));
          return;
        }
        if (view === "workflow" && selection.objectType === "workflow") {
          navigate(getSelectionPath(selection, snapshot));
          return;
        }
        if (view === "asset" && selection.objectType === "asset") {
          navigate(getSelectionPath(selection, snapshot));
          return;
        }
        if (view === "agent" && selection.objectType === "agent") {
          navigate(getSelectionPath(selection, snapshot));
          return;
        }
        if (view === "workspace" && selection.objectType === "workspace-file") {
          navigate(getSelectionPath(selection, snapshot));
          return;
        }
      }

      navigate(sectionRootByView[view]);
    },
    [navigate, selection, snapshot],
  );

  const navigateUp = useCallback((): void => {
    const parent = breadcrumbs[breadcrumbs.length - 2];
    navigate(parent?.to ?? sectionRootByView[leftPanelView]);
  }, [breadcrumbs, navigate, leftPanelView]);

  return {
    breadcrumbs,
    canNavigateUp: breadcrumbs.length > 1,
    contextStatusLabel: contextMeta.statusLabel,
    contextSubtitle: contextMeta.subtitle,
    contextTitle: contextMeta.title,
    leftPanelView,
    selection,
    navigateUp,
    setLeftPanelView,
    setSelection,
  };
};
