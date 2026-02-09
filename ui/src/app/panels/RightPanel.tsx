import { Card } from "@/components/ui/card";
import { buildRendererKeyFromSelection, renderPlanByObjectType, resolveRenderer } from "@/app/registry";
import type { InspectorTarget, Selection, WorkspaceSnapshot } from "@/app/types";

interface RightPanelProps {
  selection: Selection | null;
  snapshot: WorkspaceSnapshot;
  inspectorTarget: InspectorTarget;
  onInspectorTargetChange: (target: InspectorTarget) => void;
  onRefresh: () => void;
}

export const RightPanel = ({
  selection,
  snapshot,
  inspectorTarget,
  onInspectorTargetChange,
  onRefresh,
}: RightPanelProps): JSX.Element => {
  if (!selection) {
    return (
      <Card className="flex h-full items-center justify-center border-dashed border-border/60 bg-muted/10">
        <p className="text-sm text-muted-foreground">Inspector is idle.</p>
      </Card>
    );
  }

  const plan = renderPlanByObjectType[selection.objectType];
  const renderers = plan.right.map(target => {
    const key = buildRendererKeyFromSelection(selection, target);
    return resolveRenderer(key);
  });

  return (
    <div className="flex h-full flex-col gap-4">
      {renderers.map(renderer => (
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
