import { useCallback, useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import { AppShell } from "@/app/layout/AppShell";
import { ErrorBoundary } from "@/app/layout/ErrorBoundary";
import { OAuthCallbackPage } from "@/app/oauth/OAuthCallbackPage";
import { useWorkspaceRuns } from "@/app/runs/useWorkspaceRuns";
import { workspaceApi } from "@/app/state/api";
import { getLeftPanelViewFromPath, useNavigationState } from "@/app/state/useNavigationState";
import { useWorkspaceState } from "@/app/state/useWorkspaceState";
import type { InspectorTarget, Selection } from "@/app/types";

const buildDefaultInspectorTarget = (selection: Selection | null): InspectorTarget => {
  if (!selection) {
    return { kind: "object", objectType: "project", objectId: "" };
  }

  return {
    kind: "object",
    objectType: selection.objectType,
    objectId: selection.objectId,
  };
};

// Workspace-backed app tree. Kept as a separate component so its hooks only
// mount on non-OAuth routes — App's early return for /oauth-callback must not
// skip hooks within the same component (Rules of Hooks).
const WorkspaceApp = ({ pathname }: { pathname: string }): JSX.Element => {
  const activeView = getLeftPanelViewFromPath(pathname);
  const { snapshot, status, error, refresh } = useWorkspaceState(activeView);
  // Subscribe to the runs poller only when the user is on the runs view; the
  // hook still gives us a refresh handle even when disabled so manual refresh
  // works regardless of polling state.
  const runs = useWorkspaceRuns({ enabled: activeView === "runs" });
  const { leftPanelView, selection, setLeftPanelView, setSelection } = useNavigationState(snapshot);
  const [inspectorTarget, setInspectorTarget] = useState<InspectorTarget>(
    buildDefaultInspectorTarget(selection),
  );

  useEffect(() => {
    setInspectorTarget(buildDefaultInspectorTarget(selection));
  }, [selection]);

  if (error) {
    throw error;
  }

  const handleSelectionChange = (nextSelection: Selection): void => {
    setSelection(nextSelection);
  };

  const handleOpenWorkspace = async (
    path: string,
    options?: { createIfMissing?: boolean },
  ): Promise<void> => {
    await workspaceApi.openWorkspace(path, options?.createIfMissing ?? false);
    refresh();
  };

  const handleCreateDirectory = async (path: string): Promise<void> => {
    await workspaceApi.createDirectory(path);
    refresh();
  };

  const handleCreateFile = async (path: string): Promise<void> => {
    await workspaceApi.writeFile(path, "");
    refresh();
  };

  // The toolbar refresh button targets only the data the active view actually
  // reads — runs view pulls from the runs poller; everything else reads from
  // the workspace snapshot.
  const handleActiveRefresh = useCallback((): void => {
    if (activeView === "runs") {
      runs.refresh();
      return;
    }
    refresh();
  }, [activeView, refresh, runs]);

  const isRefreshing = activeView === "runs" ? runs.loading : status === "loading";

  return (
    <ErrorBoundary>
      <AppShell
        leftPanelView={leftPanelView}
        selection={selection}
        snapshot={snapshot}
        inspectorTarget={inspectorTarget}
        isRefreshing={isRefreshing}
        onLeftPanelViewChange={setLeftPanelView}
        onSelectionChange={handleSelectionChange}
        onInspectorTargetChange={setInspectorTarget}
        onOpenWorkspace={handleOpenWorkspace}
        onCreateDirectory={handleCreateDirectory}
        onCreateFile={handleCreateFile}
        onWorkspaceRefresh={refresh}
        onActiveRefresh={handleActiveRefresh}
      />
      {status === "loading" && (
        <div className="pointer-events-none fixed inset-0 z-50 flex items-center justify-center bg-background/50">
          <div className="rounded-md border border-border bg-background px-4 py-2 text-sm text-muted-foreground">
            Syncing workspace state...
          </div>
        </div>
      )}
    </ErrorBoundary>
  );
};

const App = (): JSX.Element => {
  const location = useLocation();
  // OAuth popup target — bypass workspace boot so the page can postMessage
  // its code/state back to the opener without spinning up the whole app.
  if (location.pathname === "/oauth-callback") {
    return <OAuthCallbackPage />;
  }
  return <WorkspaceApp pathname={location.pathname} />;
};

export default App;
