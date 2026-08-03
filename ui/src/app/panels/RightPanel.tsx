import { refFromSelection } from "@/app/entities/interop";
import { RelatedPanel } from "@/app/entities/RelatedPanel";
import {
  buildRendererKeyFromSelection,
  renderPlanByObjectType,
  resolveRenderer,
} from "@/app/registry";
import type { InspectorTarget, Selection, WorkspaceSnapshot } from "@/app/types";
import { NodeInspector } from "@/components/workbench";

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
    return <NodeInspector title="Inspector" empty emptyHint="Inspector is idle." />;
  }

  const plan = renderPlanByObjectType[selection.objectType];
  const renderers = plan.right.map((target) => {
    const key = buildRendererKeyFromSelection(selection, target);
    return resolveRenderer(key, { selection, snapshot, target });
  });

  return (
    <div className="flex h-full flex-col overflow-auto">
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
      <RelatedPanel entity={refFromSelection(selection)} snapshot={snapshot} />
    </div>
  );
};
