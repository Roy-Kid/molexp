import {
  buildRendererKeyFromSelection,
  renderPlanByObjectType,
  resolveRenderer,
} from "@/app/registry";
import { RunsPage } from "@/app/runs/RunsPage";
import { SettingsPage } from "@/app/settings/SettingsPage";
import type { InspectorTarget, LeftPanelView, Selection, WorkspaceSnapshot } from "@/app/types";

const EmptySelectionPlaceholder = (): JSX.Element => (
  <div className="flex h-full items-center justify-center p-6 text-center">
    <div className="max-w-sm space-y-2">
      <h2 className="text-base font-semibold text-foreground">Select an item to begin</h2>
      <p className="text-sm text-muted-foreground">
        Pick a project, experiment, run, or workflow from the left navigation, or open the Runs view
        to inspect every execution across the workspace.
      </p>
    </div>
  </div>
);

interface CenterPanelProps {
  selection: Selection | null;
  snapshot: WorkspaceSnapshot;
  leftPanelView?: LeftPanelView;
  inspectorTarget: InspectorTarget;

  onInspectorTargetChange: (target: InspectorTarget) => void;
  onRefresh: () => void;
}

export const CenterPanel = ({
  selection,
  snapshot,
  leftPanelView,
  inspectorTarget,
  onInspectorTargetChange,
  onRefresh,
}: CenterPanelProps): JSX.Element => {
  if (!selection) {
    if (leftPanelView === "runs") {
      return <RunsPage snapshot={snapshot} />;
    }
    if (leftPanelView === "settings") {
      return <SettingsPage />;
    }
    return <EmptySelectionPlaceholder />;
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
