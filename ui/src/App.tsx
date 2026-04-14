import { useEffect, useState } from "react";
import { AppShell } from "@/app/layout/AppShell";
import { ErrorBoundary } from "@/app/layout/ErrorBoundary";
import { workspaceApi } from "@/app/state/api";
import { useNavigationState } from "@/app/state/useNavigationState";
import { useWorkspaceState } from "@/app/state/useWorkspaceState";
import type { InspectorTarget, Selection } from "@/app/types";
import { initializeUiPlugins } from "@/plugins/runtime";

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

const App = (): JSX.Element => {
  const { snapshot, status, error, refresh } = useWorkspaceState();
  const { leftPanelView, selection, setLeftPanelView, setSelection } = useNavigationState(snapshot);
  const [pluginsReady, setPluginsReady] = useState(false);
  const [inspectorTarget, setInspectorTarget] = useState<InspectorTarget>(
    buildDefaultInspectorTarget(selection),
  );

  useEffect(() => {
    setInspectorTarget(buildDefaultInspectorTarget(selection));
  }, [selection]);

  useEffect(() => {
    let cancelled = false;

    initializeUiPlugins()
      .catch((pluginError) => {
        console.warn("Failed to initialize UI plugins:", pluginError);
      })
      .finally(() => {
        if (!cancelled) {
          setPluginsReady(true);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  if (error) {
    throw error;
  }

  if (!pluginsReady) {
    return (
      <ErrorBoundary>
        <div className="flex h-screen items-center justify-center bg-background text-foreground">
          <div className="rounded-md border border-border bg-background px-4 py-2 text-sm text-muted-foreground">
            Loading interface plugins...
          </div>
        </div>
      </ErrorBoundary>
    );
  }

  const handleSelectionChange = (nextSelection: Selection): void => {
    setSelection(nextSelection);
  };

  const handleOpenWorkspace = async (path: string): Promise<void> => {
    await workspaceApi.openWorkspace(path);
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

  return (
    <ErrorBoundary>
      <AppShell
        leftPanelView={leftPanelView}
        selection={selection}
        snapshot={snapshot}
        inspectorTarget={inspectorTarget}
        onLeftPanelViewChange={setLeftPanelView}
        onSelectionChange={handleSelectionChange}
        onInspectorTargetChange={setInspectorTarget}
        onOpenWorkspace={handleOpenWorkspace}
        onCreateDirectory={handleCreateDirectory}
        onCreateFile={handleCreateFile}
        onWorkspaceRefresh={refresh}
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

export default App;
