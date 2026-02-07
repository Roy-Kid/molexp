import { useState } from "react";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import type { InspectorTarget, Selection, WorkspaceSnapshot } from "@/app/types";
import { TopBar } from "@/app/layout/TopBar";
import { LeftPanel } from "@/app/panels/LeftPanel";
import { CenterPanel } from "@/app/panels/CenterPanel";
import type { LeftPanelView } from "@/app/types";

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
  left: 20,
  center: 80,
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

  return (
    <div className="flex h-screen flex-col bg-background text-foreground">
      <TopBar searchQuery={searchQuery} onSearchChange={setSearchQuery} />
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
        </ResizablePanelGroup>
      </main>
    </div>
  );
};
