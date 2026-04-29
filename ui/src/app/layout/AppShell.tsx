import { PanelRightClose, PanelRightOpen } from "lucide-react";
import { useState } from "react";
import { ContextBar } from "@/app/layout/ContextBar";
import { CenterPanel } from "@/app/panels/CenterPanel";
import { LeftPanel } from "@/app/panels/LeftPanel";
import { RightPanel } from "@/app/panels/RightPanel";
import type { InspectorTarget, LeftPanelView, Selection, WorkspaceSnapshot } from "@/app/types";
import { Button } from "@/components/ui/button";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface AppShellProps {
  leftPanelView: LeftPanelView;
  selection: Selection | null;
  snapshot: WorkspaceSnapshot;
  inspectorTarget: InspectorTarget;
  isRefreshing: boolean;
  onLeftPanelViewChange: (view: LeftPanelView) => void;
  onSelectionChange: (selection: Selection) => void;
  onInspectorTargetChange: (target: InspectorTarget) => void;
  onOpenWorkspace: (path: string) => void;
  onCreateDirectory: (path: string) => void;
  onCreateFile: (path: string) => void;
  onWorkspaceRefresh: () => void;
  onActiveRefresh: () => void;
}

const NAV_SIZE = { default: 22, min: 16, max: 30 };
const INSPECTOR_SIZE = { default: 30, min: 20, max: 45 };

export const AppShell = ({
  leftPanelView,
  selection,
  snapshot,
  inspectorTarget,
  isRefreshing,
  onLeftPanelViewChange,
  onSelectionChange,
  onInspectorTargetChange,
  onOpenWorkspace,
  onCreateDirectory,
  onCreateFile,
  onWorkspaceRefresh,
  onActiveRefresh,
}: AppShellProps): JSX.Element => {
  const [searchQuery, setSearchQuery] = useState("");
  const [inspectorOpen, setInspectorOpen] = useState(false);

  const inspectorVisible = inspectorOpen && Boolean(selection);
  const toggleDisabled = !selection;
  const toggleLabel = inspectorVisible ? "Hide details" : "Show details";

  return (
    <div className="flex h-screen flex-col bg-background text-foreground">
      <ContextBar
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        onRefresh={onActiveRefresh}
        isRefreshing={isRefreshing}
      />
      <main className="flex flex-1 flex-col overflow-hidden">
        <ResizablePanelGroup direction="horizontal" className="flex-1">
          <ResizablePanel
            defaultSize={NAV_SIZE.default}
            minSize={NAV_SIZE.min}
            maxSize={NAV_SIZE.max}
          >
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
          <ResizablePanel defaultSize={100 - NAV_SIZE.default}>
            <ResizablePanelGroup direction="horizontal" className="h-full">
              <ResizablePanel defaultSize={inspectorVisible ? 100 - INSPECTOR_SIZE.default : 100}>
                <div className="flex h-full flex-col">
                  <div className="flex h-9 items-center justify-end border-b border-border/70 bg-muted/10 px-2">
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-7 w-7"
                            onClick={() => setInspectorOpen((current) => !current)}
                            disabled={toggleDisabled}
                            aria-label={toggleLabel}
                          >
                            {inspectorVisible ? (
                              <PanelRightClose className="h-4 w-4" />
                            ) : (
                              <PanelRightOpen className="h-4 w-4" />
                            )}
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent side="left">{toggleLabel}</TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </div>
                  <div className="flex-1 overflow-hidden">
                    <CenterPanel
                      selection={selection}
                      snapshot={snapshot}
                      leftPanelView={leftPanelView}
                      inspectorTarget={inspectorTarget}
                      onInspectorTargetChange={onInspectorTargetChange}
                      onRefresh={onWorkspaceRefresh}
                    />
                  </div>
                </div>
              </ResizablePanel>
              {inspectorVisible && (
                <>
                  <ResizableHandle withHandle />
                  <ResizablePanel
                    defaultSize={INSPECTOR_SIZE.default}
                    minSize={INSPECTOR_SIZE.min}
                    maxSize={INSPECTOR_SIZE.max}
                  >
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
          </ResizablePanel>
        </ResizablePanelGroup>
      </main>
    </div>
  );
};
