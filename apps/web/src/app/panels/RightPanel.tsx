import { CopilotPanel } from "@/app/components/CopilotPanel";
import { refFromSelection } from "@/app/entities/interop";
import { RelatedPanel } from "@/app/entities/RelatedPanel";
import {
  buildRendererKeyFromSelection,
  renderPlanByObjectType,
  tryResolveRenderer,
} from "@/app/registry";
import type { InspectorTarget, Selection, WorkspaceSnapshot } from "@/app/types";
import { NodeInspector } from "@/components/workbench";
import { usePluginPreferencesGeneration } from "@/plugins/preferences";

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
  usePluginPreferencesGeneration();

  if (!selection) {
    return (
      <div className="flex h-full flex-col overflow-auto">
        <CopilotPanel snapshot={snapshot} />
        <NodeInspector title="Inspector" empty emptyHint="Select an entity for details." />
      </div>
    );
  }

  const plan = renderPlanByObjectType[selection.objectType];
  const renderers = plan.right
    .map((target) => {
      const key = buildRendererKeyFromSelection(selection, target);
      return tryResolveRenderer(key, { selection, snapshot, target });
    })
    .filter((renderer): renderer is NonNullable<typeof renderer> => renderer !== null);

  return (
    <div className="flex h-full flex-col overflow-auto">
      <CopilotPanel snapshot={snapshot} />
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
