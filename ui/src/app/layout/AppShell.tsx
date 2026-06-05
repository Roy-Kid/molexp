import { PanelRightClose, PanelRightOpen } from "lucide-react";
import { useCallback, useMemo, useState } from "react";
import { Breadcrumb } from "@/app/entities/Breadcrumb";
import { buildTrail } from "@/app/entities/breadcrumb";
import { GlobalCommandPalette } from "@/app/entities/GlobalCommandPalette";
import { ContextBar } from "@/app/layout/ContextBar";
import { CenterPanel } from "@/app/panels/CenterPanel";
import { LeftPanel } from "@/app/panels/LeftPanel";
import { RightPanel } from "@/app/panels/RightPanel";
import { type InspectedTask, InspectedTaskContext } from "@/app/state/inspectedTask";
import type { InspectorTarget, LeftPanelView, Selection, WorkspaceSnapshot } from "@/app/types";
import { Button } from "@/components/ui/button";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { useIsMobile } from "@/hooks/use-is-mobile";

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
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [inspectedTask, setInspectedTask] = useState<InspectedTask | null>(null);
  const isMobile = useIsMobile();

  const inspectTask = useCallback((taskId: string, runId: string): void => {
    // Pin the node to the right inspector and open the panel in-place — never
    // navigate. This is what makes a node click expand the sidebar instead of
    // jumping to a standalone task page.
    setInspectedTask({ taskId, runId });
    setInspectorOpen(true);
  }, []);

  const clearInspectedTask = useCallback((): void => {
    setInspectedTask(null);
  }, []);

  const inspectedTaskContext = useMemo(
    () => ({ inspectedTask, inspectTask, clearInspectedTask }),
    [inspectedTask, inspectTask, clearInspectedTask],
  );

  // A pinned task only applies while we are still on the run it was opened
  // from; navigating to any other object drops back to that object's own
  // inspector. Checking validity at render time (rather than clearing via an
  // effect) keeps the pin self-invalidating with no extra re-render.
  const pinnedTaskActive =
    inspectedTask !== null &&
    ((selection?.objectType === "run" && selection.objectId === inspectedTask.runId) ||
      // Preview (compiled, un-run) workflows are inspected from the workflow
      // entity's Graph tab — there is no run to scope the pin to.
      selection?.objectType === "workflow");

  // The right inspector shows the pinned task when one is active, otherwise the
  // page's own object. The synthetic `task` Selection is never routed to the
  // URL — it only lets the renderer registry resolve the task inspector.
  const inspectorSelection: Selection | null =
    inspectedTask && pinnedTaskActive
      ? {
          objectType: "task",
          taskId: inspectedTask.taskId,
          runId: inspectedTask.runId,
          objectId: inspectedTask.taskId,
        }
      : selection;

  const inspectorVisible = inspectorOpen && Boolean(inspectorSelection);
  const toggleDisabled = !inspectorSelection;
  const toggleLabel = inspectorVisible ? "Hide details" : "Show details";

  const trail = useMemo(
    () => buildTrail(selection, leftPanelView, snapshot),
    [selection, leftPanelView, snapshot],
  );

  // On mobile the nav drawer is dismissed once a selection is made so the
  // freshly-selected object is visible in the full-width center pane.
  const handleNavSelect = useCallback(
    (next: Selection): void => {
      onSelectionChange(next);
      setMobileNavOpen(false);
    },
    [onSelectionChange],
  );

  const navContent = (
    <LeftPanel
      view={leftPanelView}
      selection={selection}
      snapshot={snapshot}
      searchQuery={searchQuery}
      onViewChange={onLeftPanelViewChange}
      onSelect={isMobile ? handleNavSelect : onSelectionChange}
      onOpenWorkspace={onOpenWorkspace}
      onCreateDirectory={onCreateDirectory}
      onCreateFile={onCreateFile}
      onRefresh={onWorkspaceRefresh}
    />
  );

  const inspectorToggle = (
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
  );

  const centerContent = (
    <div className="flex h-full flex-col">
      <div className="flex h-9 items-center justify-between gap-2 border-b border-border/70 bg-muted/10 px-3">
        <Breadcrumb items={trail} />
        {inspectorToggle}
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
  );

  const inspectorContent = (
    <RightPanel
      selection={inspectorSelection}
      snapshot={snapshot}
      inspectorTarget={inspectorTarget}
      onInspectorTargetChange={onInspectorTargetChange}
      onRefresh={onWorkspaceRefresh}
    />
  );

  return (
    <InspectedTaskContext.Provider value={inspectedTaskContext}>
      <GlobalCommandPalette snapshot={snapshot} />
      <div className="flex h-screen flex-col bg-background text-foreground">
        <ContextBar
          searchQuery={searchQuery}
          onSearchChange={setSearchQuery}
          onRefresh={onActiveRefresh}
          isRefreshing={isRefreshing}
          onMenuClick={isMobile ? () => setMobileNavOpen(true) : undefined}
        />
        <main className="flex flex-1 flex-col overflow-hidden">
          {isMobile ? (
            // Small screens: a single full-width center pane. The nav and the
            // inspector each move into an edge drawer so neither is squeezed to
            // an unusable width by the fixed-percentage desktop split.
            <>
              <div className="flex-1 overflow-hidden">{centerContent}</div>
              <Sheet open={mobileNavOpen} onOpenChange={setMobileNavOpen}>
                <SheetContent side="left" className="w-[85vw] max-w-sm p-0">
                  <SheetHeader className="sr-only">
                    <SheetTitle>Navigation</SheetTitle>
                    <SheetDescription>Workspace tree and views</SheetDescription>
                  </SheetHeader>
                  <div className="h-full overflow-hidden">{navContent}</div>
                </SheetContent>
              </Sheet>
              <Sheet open={inspectorVisible} onOpenChange={setInspectorOpen}>
                <SheetContent side="right" className="w-[85vw] max-w-md p-0">
                  <SheetHeader className="sr-only">
                    <SheetTitle>Inspector</SheetTitle>
                    <SheetDescription>Details for the selected object</SheetDescription>
                  </SheetHeader>
                  <div className="h-full overflow-hidden bg-muted/10">{inspectorContent}</div>
                </SheetContent>
              </Sheet>
            </>
          ) : (
            <ResizablePanelGroup direction="horizontal" className="flex-1">
              <ResizablePanel
                defaultSize={NAV_SIZE.default}
                minSize={NAV_SIZE.min}
                maxSize={NAV_SIZE.max}
              >
                {navContent}
              </ResizablePanel>
              <ResizableHandle withHandle />
              <ResizablePanel defaultSize={100 - NAV_SIZE.default}>
                <ResizablePanelGroup direction="horizontal" className="h-full">
                  <ResizablePanel
                    defaultSize={inspectorVisible ? 100 - INSPECTOR_SIZE.default : 100}
                  >
                    {centerContent}
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
                          {inspectorContent}
                        </div>
                      </ResizablePanel>
                    </>
                  )}
                </ResizablePanelGroup>
              </ResizablePanel>
            </ResizablePanelGroup>
          )}
        </main>
      </div>
    </InspectedTaskContext.Provider>
  );
};
