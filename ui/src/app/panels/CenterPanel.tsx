import {
  buildRendererKeyFromSelection,
  renderPlanByObjectType,
  resolveRenderer,
} from "@/app/registry";
import type { InspectorTarget, Selection, WorkspaceSnapshot } from "@/app/types";
import { Card } from "@/components/ui/card";

interface CenterPanelProps {
  selection: Selection | null;
  snapshot: WorkspaceSnapshot;
  inspectorTarget: InspectorTarget;

  onInspectorTargetChange: (target: InspectorTarget) => void;
  onRefresh: () => void;
}

export const CenterPanel = ({
  selection,
  snapshot,
  inspectorTarget,
  onInspectorTargetChange,
  onRefresh,
}: CenterPanelProps): JSX.Element => {
  if (!selection) {
    return (
      <Card className="flex h-full items-center justify-center border-dashed border-border/60 bg-muted/20">
        <p className="text-sm text-muted-foreground">
          Select a project, workflow, or asset to begin.
        </p>
      </Card>
    );
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
