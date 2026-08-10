import { PanelRightClose, PanelRightOpen } from "lucide-react";
import { useCallback, useMemo, useRef, useState } from "react";
import { Breadcrumb } from "@/app/entities/Breadcrumb";
import { buildTrail } from "@/app/entities/breadcrumbTrail";
import { GlobalCommandPalette } from "@/app/entities/GlobalCommandPalette";
import { ContextBar } from "@/app/layout/ContextBar";
import { CenterPanel } from "@/app/panels/CenterPanel";
import { LeftPanel } from "@/app/panels/LeftPanel";
import { RightPanel } from "@/app/panels/RightPanel";
import { RunInspector, type RunInspectorRegistration } from "@/app/runs/inspector/RunInspector";
import { type InspectedTask, InspectedTaskContext } from "@/app/state/inspectedTask";
import type { InspectorTarget, LeftPanelView, Selection, WorkspaceSnapshot } from "@/app/types";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { WorkbenchStatusStrip, WorkbenchToggleAction } from "@/components/workbench";
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
  onOpenWorkspace: (path: string, options?: { createIfMissing?: boolean }) => Promise<void>;
  onCreateDirectory: (path: string) => void;
  onCreateFile: (path: string) => void;
  onExpandDirectory?: (dirPath: string) => void;
  onExpandProject?: (projectId: string) => void;
  onExpandExperiment?: (projectId: string, experimentId: string) => void;
  isProjectExpanded?: (projectId: string) => boolean;
  isExperimentExpanded?: (projectId: string, experimentId: string) => boolean;
  /** Bumps after refresh clears lazy entity caches — TreeView re-hydrates open folders. */
  dataEpoch?: number;
  onWorkspaceRefresh: () => void;
  onActiveRefresh: () => void;
}

const NAV_SIZE = { default: 22, min: 16, max: 30 };
const INSPECTOR_SIZE = { default: 30, min: 20, max: 45 };
const SHELL_PANEL_IDS = ["navigator", "workspace"];

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
  onExpandDirectory,
  onExpandProject,
  onExpandExperiment,
  isProjectExpanded,
  isExperimentExpanded,
  dataEpoch = 0,
  onWorkspaceRefresh,
  onActiveRefresh,
}: AppShellProps): JSX.Element => {
  const [searchQuery, setSearchQuery] = useState("");
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [inspectedTask, setInspectedTask] = useState<InspectedTask | null>(null);
  const [runInspector, setRunInspector] = useState<RunInspectorRegistration | null>(null);
  const registeredRunId = useRef<string | null>(null);
  const isMobile = useIsMobile();

  const inspectTask = useCallback((taskId: string, runId: string): void => {
    setInspectedTask({ taskId, runId });
    setInspectorOpen(true);
  }, []);

  const clearInspectedTask = useCallback((): void => {
    setInspectedTask(null);
  }, []);

  const handleRunInspectorChange = useCallback(
    (registration: RunInspectorRegistration | null): void => {
      const nextRunId = registration?.run?.id ?? null;
      setRunInspector(registration);
      if (nextRunId && nextRunId !== registeredRunId.current) {
        setInspectorOpen(true);
      }
      registeredRunId.current = nextRunId;
    },
    [],
  );

  const inspectedTaskContext = useMemo(
    () => ({ inspectedTask, inspectTask, clearInspectedTask }),
    [inspectedTask, inspectTask, clearInspectedTask],
  );

  const pinnedTaskActive =
    inspectedTask !== null &&
    ((selection?.objectType === "run" && selection.objectId === inspectedTask.runId) ||
      selection?.objectType === "workflow" ||
      selection?.objectType === "experiment");

  const inspectorSelection: Selection | null =
    inspectedTask && pinnedTaskActive
      ? {
          objectType: "task",
          taskId: inspectedTask.taskId,
          runId: inspectedTask.runId,
          objectId: inspectedTask.taskId,
        }
      : selection;

  const hasInspectorContent = Boolean(runInspector || inspectorSelection);
  const inspectorVisible = inspectorOpen && hasInspectorContent;
  const toggleDisabled = !hasInspectorContent;
  const toggleLabel = inspectorVisible ? "Hide details" : "Show details";
  const inspectorPanelIds = useMemo(
    () => (inspectorVisible ? ["work-surface", "inspector"] : ["work-surface"]),
    [inspectorVisible],
  );

  const trail = useMemo(
    () => buildTrail(selection, leftPanelView, snapshot),
    [selection, leftPanelView, snapshot],
  );

  const activeWorkspace = useMemo(
    () => snapshot.workspaces.find((w) => w.active) ?? snapshot.workspaces[0] ?? null,
    [snapshot.workspaces],
  );

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
      onExpandDirectory={onExpandDirectory}
      onExpandProject={onExpandProject}
      onExpandExperiment={onExpandExperiment}
      isProjectExpanded={isProjectExpanded}
      isExperimentExpanded={isExperimentExpanded}
      dataEpoch={dataEpoch}
      onRefresh={onWorkspaceRefresh}
    />
  );

  const inspectorToggle = (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <WorkbenchToggleAction
            label={toggleLabel}
            pressed={inspectorVisible}
            disabled={toggleDisabled}
            onClick={() => setInspectorOpen((current) => !current)}
          >
            {inspectorVisible ? (
              <PanelRightClose className="h-4 w-4" />
            ) : (
              <PanelRightOpen className="h-4 w-4" />
            )}
          </WorkbenchToggleAction>
        </TooltipTrigger>
        <TooltipContent side="left">{toggleLabel}</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );

  const centerContent = (
    <div className="flex h-full min-h-0 flex-col">
      {/* Work-surface header: breadcrumb left, primary chrome right — 40px */}
      <div className="flex h-10 flex-none items-center justify-between gap-2 border-b border-border bg-surface px-3">
        <Breadcrumb items={trail} />
        {inspectorToggle}
      </div>
      <div className="min-h-0 flex-1 overflow-hidden bg-canvas">
        <CenterPanel
          selection={selection}
          snapshot={snapshot}
          leftPanelView={leftPanelView}
          inspectorTarget={inspectorTarget}
          onInspectorTargetChange={onInspectorTargetChange}
          onRunInspectorChange={handleRunInspectorChange}
          onRefresh={onWorkspaceRefresh}
        />
      </div>
    </div>
  );

  const inspectorContent = runInspector ? (
    <RunInspector {...runInspector} className="border-l-0 bg-surface-subtle" />
  ) : (
    <RightPanel
      selection={inspectorSelection}
      snapshot={snapshot}
      inspectorTarget={inspectorTarget}
      onInspectorTargetChange={onInspectorTargetChange}
      onRefresh={onWorkspaceRefresh}
    />
  );

  const workbenchColumns = isMobile ? (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
      <div className="min-h-0 flex-1 overflow-hidden">{centerContent}</div>
      <Sheet open={mobileNavOpen} onOpenChange={setMobileNavOpen}>
        <SheetContent side="left" className="w-dialog-viewport max-w-sm p-0">
          <SheetHeader className="sr-only">
            <SheetTitle>Navigation</SheetTitle>
            <SheetDescription>Workspace tree and views</SheetDescription>
          </SheetHeader>
          <div className="h-full overflow-hidden">{navContent}</div>
        </SheetContent>
      </Sheet>
      <Sheet open={inspectorVisible} onOpenChange={setInspectorOpen}>
        <SheetContent side="right" className="w-dialog-viewport max-w-md p-0">
          <SheetHeader className="sr-only">
            <SheetTitle>Inspector</SheetTitle>
            <SheetDescription>Details for the selected object</SheetDescription>
          </SheetHeader>
          <div className="h-full overflow-hidden bg-surface-subtle">{inspectorContent}</div>
        </SheetContent>
      </Sheet>
    </div>
  ) : (
    <ResizablePanelGroup
      id="molexp-workbench-shell"
      direction="horizontal"
      autoSaveId="molexp.workbench.shell"
      autoSavePanelIds={SHELL_PANEL_IDS}
      className="min-h-0 flex-1"
    >
      <ResizablePanel
        id="navigator"
        defaultSize={NAV_SIZE.default}
        minSize={NAV_SIZE.min}
        maxSize={NAV_SIZE.max}
      >
        <div className="h-full overflow-hidden border-r border-border bg-surface">{navContent}</div>
      </ResizablePanel>
      <ResizableHandle withHandle />
      <ResizablePanel id="workspace" defaultSize={100 - NAV_SIZE.default}>
        <ResizablePanelGroup
          id="molexp-workbench-detail"
          direction="horizontal"
          autoSaveId="molexp.workbench.detail"
          autoSavePanelIds={inspectorPanelIds}
          className="h-full"
        >
          <ResizablePanel
            id="work-surface"
            defaultSize={inspectorVisible ? 100 - INSPECTOR_SIZE.default : 100}
          >
            {centerContent}
          </ResizablePanel>
          {inspectorVisible && (
            <>
              <ResizableHandle withHandle />
              <ResizablePanel
                id="inspector"
                defaultSize={INSPECTOR_SIZE.default}
                minSize={INSPECTOR_SIZE.min}
                maxSize={INSPECTOR_SIZE.max}
              >
                <div className="mol-motion-enter-from-right h-full overflow-hidden border-l border-border bg-surface-subtle">
                  {inspectorContent}
                </div>
              </ResizablePanel>
            </>
          )}
        </ResizablePanelGroup>
      </ResizablePanel>
    </ResizablePanelGroup>
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
          activeWorkspace={activeWorkspace}
        />
        {/* Work surface above a full-width MolVis-style status bar. */}
        <main className="flex min-h-0 flex-1 flex-col overflow-hidden">
          {workbenchColumns}
          <WorkbenchStatusStrip
            isRefreshing={isRefreshing}
            onRemoteIndexReady={onWorkspaceRefresh}
            activeWorkspace={activeWorkspace}
          />
        </main>
      </div>
    </InspectedTaskContext.Provider>
  );
};
