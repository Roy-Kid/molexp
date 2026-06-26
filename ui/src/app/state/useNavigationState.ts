import { useCallback, useMemo } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import type { LeftPanelView, ObjectView, Selection, WorkspaceSnapshot } from "@/app/types";

const sectionRootByView: Record<LeftPanelView, string> = {
  projects: "/projects",
  workspace: "/workspace",
  runs: "/runs",
  workflow: "/workflows",
  asset: "/assets",
  agent: "/agent-tasks",
  knowledge: "/knowledge",
  settings: "/settings",
};

const buildWorkspaceFileSelection = (searchParams: URLSearchParams): Selection | null => {
  const filePath = searchParams.get("file");
  const fileKind = searchParams.get("fileKind");

  if (!filePath || !fileKind) {
    return null;
  }

  const assetId = searchParams.get("assetId");
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
    assetId: assetId ?? undefined,
    hasPreviewSidecar: searchParams.get("hasPreviewSidecar") === "1",
  };
};

export const getLeftPanelViewFromPath = (pathname: string): LeftPanelView => {
  if (pathname.startsWith("/workspace")) {
    return "workspace";
  }
  if (pathname.startsWith("/runs")) {
    return "runs";
  }
  if (pathname.startsWith("/workflows")) {
    return "workflow";
  }
  if (pathname.startsWith("/assets")) {
    return "asset";
  }
  if (pathname.startsWith("/agent-tasks")) {
    return "agent";
  }
  if (pathname.startsWith("/knowledge")) {
    return "knowledge";
  }
  if (pathname.startsWith("/settings")) {
    return "settings";
  }
  return "projects";
};

const parseObjectView = (raw: string | null): ObjectView | undefined => {
  if (raw === "overview" || raw === "logs" || raw === "metrics" || raw === "scheduler") {
    return raw;
  }
  return undefined;
};

const buildSelectionFromLocation = (
  pathname: string,
  searchParams: URLSearchParams,
): Selection | null => {
  const objectView = parseObjectView(searchParams.get("tab"));

  const taskMatch = pathname.match(
    /^\/projects\/([^/]+)\/experiments\/([^/]+)\/runs\/([^/]+)\/tasks\/([^/]+)$/,
  );
  if (taskMatch) {
    return {
      objectType: "task",
      taskId: decodeURIComponent(taskMatch[4]),
      runId: decodeURIComponent(taskMatch[3]),
      objectId: decodeURIComponent(taskMatch[4]),
    };
  }

  const projectRunMatch = pathname.match(
    /^\/projects\/([^/]+)\/experiments\/([^/]+)\/runs\/([^/]+)$/,
  );
  if (projectRunMatch) {
    return {
      objectType: "run",
      objectId: decodeURIComponent(projectRunMatch[3]),
      objectView,
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

  if (pathname === "/agent-tasks/new") {
    return {
      objectType: "agent",
      objectId: "new",
    };
  }

  const agentMatch = pathname.match(/^\/agent-tasks\/([^/]+)$/);
  if (agentMatch) {
    return {
      objectType: "agent",
      objectId: decodeURIComponent(agentMatch[1]),
    };
  }

  if (pathname === "/knowledge" || pathname.startsWith("/knowledge/")) {
    // The concept's bundle-relative path (which may contain "/") is the rest
    // after "/knowledge/"; bare "/knowledge" is the browse overview.
    const rest = pathname === "/knowledge" ? "" : pathname.slice("/knowledge/".length);
    return { objectType: "knowledge", objectId: decodeURIComponent(rest) };
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
      const path = `/projects/${encodeURIComponent(run.projectId)}/experiments/${encodeURIComponent(run.experimentId)}/runs/${encodeURIComponent(run.id)}`;
      if (!selection.objectView) {
        return path;
      }
      const params = new URLSearchParams({ tab: selection.objectView });
      return `${path}?${params.toString()}`;
    }
    case "task": {
      const run = snapshot.runs.find((item) => item.id === selection.runId);
      if (!run) {
        return "/projects";
      }
      return `/projects/${encodeURIComponent(run.projectId)}/experiments/${encodeURIComponent(run.experimentId)}/runs/${encodeURIComponent(run.id)}/tasks/${encodeURIComponent(selection.taskId)}`;
    }
    case "workflow":
      return `/workflows/${encodeURIComponent(selection.workflowId)}`;
    case "asset":
      return `/assets/${encodeURIComponent(selection.objectId)}`;
    case "agent":
      return selection.objectId === "new"
        ? "/agent-tasks/new"
        : `/agent-tasks/${encodeURIComponent(selection.objectId)}`;
    case "knowledge":
      // objectId is a bundle-relative path (may contain "/"); keep the slashes
      // readable in the URL by encoding each segment, not the whole string.
      return selection.objectId
        ? `/knowledge/${selection.objectId.split("/").map(encodeURIComponent).join("/")}`
        : "/knowledge";
    case "workspace-file": {
      const params = new URLSearchParams({
        file: selection.filePath,
        fileKind: selection.fileKind,
      });
      if (selection.assetId) {
        params.set("assetId", selection.assetId);
      }
      if (selection.hasPreviewSidecar) {
        params.set("hasPreviewSidecar", "1");
      }
      return `/workspace?${params.toString()}`;
    }
  }
};

export interface NavigationState {
  leftPanelView: LeftPanelView;
  selection: Selection | null;
  setLeftPanelView: (view: LeftPanelView) => void;
  setSelection: (selection: Selection | null) => void;
}

export const useNavigationState = (snapshot: WorkspaceSnapshot): NavigationState => {
  const location = useLocation();
  const navigate = useNavigate();

  const searchParams = useMemo(() => new URLSearchParams(location.search), [location.search]);

  const leftPanelView = useMemo(
    () => getLeftPanelViewFromPath(location.pathname),
    [location.pathname],
  );

  const selection = useMemo(
    () => buildSelectionFromLocation(location.pathname, searchParams),
    [location.pathname, searchParams],
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
        if (
          view === "projects" &&
          ["project", "experiment", "run"].includes(selection.objectType)
        ) {
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

  return {
    leftPanelView,
    selection,
    setLeftPanelView,
    setSelection,
  };
};
