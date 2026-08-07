import {
  buildRendererKeyFromSelection,
  renderPlanByObjectType,
  resolveRenderer,
} from "@/app/registry";
import { ActivityPage } from "@/app/runs/ActivityPage";
import type { RunInspectorRegistration } from "@/app/runs/inspector/RunInspector";
import { RunsPage } from "@/app/runs/RunsPage";
import { SettingsPage } from "@/app/settings/SettingsPage";
import type { InspectorTarget, LeftPanelView, Selection, WorkspaceSnapshot } from "@/app/types";
import { WorkflowsPage } from "@/app/workflows/WorkflowsPage";
import { WorkbenchOperationState } from "@/components/workbench";

interface EmptySelectionCopy {
  title: string;
  description: string;
}

// Per-view empty-selection copy — the placeholder must speak the language of
// the section the user is looking at, not always the Experiments tree.
const EMPTY_SELECTION_COPY: Partial<Record<LeftPanelView, EmptySelectionCopy>> = {
  agent: {
    title: "No agent task selected",
    description: "Select an agent task from the left, or start a new one.",
  },
  knowledge: {
    title: "No document selected",
    description: "Pick a note from the left, or create a new one.",
  },
  activity: {
    title: "Workspace activity",
    description: "The global event timeline fills the center panel for this section.",
  },
};

const DEFAULT_EMPTY_SELECTION_COPY: EmptySelectionCopy = {
  title: "Select an item to begin",
  description:
    "Pick a project, experiment, run, or workflow from the left navigation, or open the Runs " +
    "view to inspect every execution across the workspace.",
};

/** Resolve the placeholder copy for a left-panel view (exported for tests). */
export const emptySelectionCopy = (view?: LeftPanelView): EmptySelectionCopy =>
  (view && EMPTY_SELECTION_COPY[view]) || DEFAULT_EMPTY_SELECTION_COPY;

const EmptySelectionPlaceholder = ({ view }: { view?: LeftPanelView }): JSX.Element => {
  const copy = emptySelectionCopy(view);
  return (
    <div className="flex h-full items-center justify-center p-6">
      <WorkbenchOperationState kind="empty" title={copy.title} detail={copy.description} />
    </div>
  );
};

interface CenterPanelProps {
  selection: Selection | null;
  snapshot: WorkspaceSnapshot;
  leftPanelView?: LeftPanelView;
  inspectorTarget: InspectorTarget;

  onInspectorTargetChange: (target: InspectorTarget) => void;
  onRunInspectorChange: (registration: RunInspectorRegistration | null) => void;
  onRefresh: () => void;
}

export const CenterPanel = ({
  selection,
  snapshot,
  leftPanelView,
  inspectorTarget,
  onInspectorTargetChange,
  onRunInspectorChange,
  onRefresh,
}: CenterPanelProps): JSX.Element => {
  if (!selection) {
    if (leftPanelView === "runs") {
      return <RunsPage snapshot={snapshot} onInspectorChange={onRunInspectorChange} />;
    }
    if (leftPanelView === "activity") {
      return <ActivityPage snapshot={snapshot} />;
    }
    if (leftPanelView === "workflow") {
      return <WorkflowsPage snapshot={snapshot} onRefresh={onRefresh} />;
    }
    if (leftPanelView === "settings") {
      return <SettingsPage />;
    }
    return <EmptySelectionPlaceholder view={leftPanelView} />;
  }

  const plan = renderPlanByObjectType[selection.objectType];
  const renderers = plan.center.map((target) => {
    const key = buildRendererKeyFromSelection(selection, target);
    return resolveRenderer(key, { selection, snapshot, target });
  });

  return (
    <div className="flex h-full flex-col">
      {renderers.map((renderer) => (
        <renderer.Component
          key={`${renderer.title}-${renderer.panelSlot}`}
          selection={selection}
          snapshot={snapshot}
          inspectorTarget={inspectorTarget}
          onInspectorTargetChange={onInspectorTargetChange}
          onRefresh={onRefresh}
        />
      ))}
    </div>
  );
};
