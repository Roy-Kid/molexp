import { useState } from "react";
import { ContextBar } from "@/app/layout/ContextBar";
import { CenterPanel } from "@/app/panels/CenterPanel";
import { LeftPanel } from "@/app/panels/LeftPanel";
import { RightPanel } from "@/app/panels/RightPanel";
import type { InspectorTarget, LeftPanelView, Selection, WorkspaceSnapshot } from "@/app/types";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";

interface AppShellProps {
  leftPanelView: LeftPanelView;
  selection: Selection | null;
  snapshot: WorkspaceSnapshot;
  inspectorTarget: InspectorTarget;
  onLeftPanelViewChange: (view: LeftPanelView) => void;
  onSelectionChange: (selection: Selection) => void;
  onInspectorTargetChange: (target: InspectorTarget) => void;
  onOpenWorkspace: (path: string) => void;
  onCreateDirectory: (path: string) => void;
  onCreateFile: (path: string) => void;
  onWorkspaceRefresh: () => void;
}

const PANEL_DEFAULTS = {
  left: 22,
  center: 78,
  right: 20,
};

export const AppShell = ({
  leftPanelView,
  selection,
  snapshot,
  inspectorTarget,
  onLeftPanelViewChange,
  onSelectionChange,
  onInspectorTargetChange,
  onOpenWorkspace,
  onCreateDirectory,
  onCreateFile,
  onWorkspaceRefresh,
}: AppShellProps): JSX.Element => {
  const [searchQuery, setSearchQuery] = useState("");
  const [inspectorOpen, setInspectorOpen] = useState(false);

  return (
    <div className="flex h-screen flex-col bg-background text-foreground">
      <ContextBar
        inspectorOpen={inspectorOpen}
        searchQuery={searchQuery}
        selectionActive={Boolean(selection)}
        onSearchChange={setSearchQuery}
        onToggleInspector={() => setInspectorOpen((current) => !current)}
      />
      <main className="flex flex-1 flex-col overflow-hidden">
        <ResizablePanelGroup direction="horizontal" className="flex-1">
          <ResizablePanel defaultSize={PANEL_DEFAULTS.left} minSize={16} maxSize={30}>
            <LeftPanel
              view={leftPanelView}
              selection={selection}
              snapshot={snapshot}
              searchQuery={searchQuery}
              onViewChange={onLeftPanelViewChange}
              onSelect={onSelectionChange}
              onOpenWorkspace={onOpenWorkspace}
              onCreateDirectory={onCreateDirectory}
              onCreateFile={onCreateFile}
              onRefresh={onWorkspaceRefresh}
            />
          </ResizablePanel>
          <ResizableHandle withHandle />
          <ResizablePanel defaultSize={PANEL_DEFAULTS.center}>
            <div className="h-full overflow-hidden">
              <CenterPanel
                selection={selection}
                snapshot={snapshot}
                inspectorTarget={inspectorTarget}
                onInspectorTargetChange={onInspectorTargetChange}
                onRefresh={onWorkspaceRefresh}
              />
            </div>
          </ResizablePanel>
          {inspectorOpen && selection && (
            <>
              <ResizableHandle withHandle />
              <ResizablePanel defaultSize={PANEL_DEFAULTS.right} minSize={16} maxSize={28}>
                <div className="h-full overflow-hidden border-l border-border/70 bg-muted/10">
                  <RightPanel
                    selection={selection}
                    snapshot={snapshot}
                    inspectorTarget={inspectorTarget}
                    onInspectorTargetChange={onInspectorTargetChange}
                    onRefresh={onWorkspaceRefresh}
                  />
                </div>
              </ResizablePanel>
            </>
          )}
        </ResizablePanelGroup>
      </main>
    </div>
  );
};
