/**
 * FlowgramCanvas — the single read-only workflow canvas, built on the
 * `@flowgram.ai/free-layout-editor` free-layout core.
 *
 * Every workflow surface (the run "what ran" preview, the workflow Graph tab,
 * the workspace `workflow.json` viewer) renders through this one component.
 * Nodes are drawn with molexp's own shadcn/ui + Tailwind chrome (NO FlowGram
 * form-materials / Semi Design / Ant Design). The canvas is strictly read-only;
 * editing / connecting / writing back is spec 04.
 */

import {
  EditorRenderer,
  FreeLayoutEditorProvider,
  type FreeLayoutProps,
  useAutoLayout,
  useNodeRender,
  type WorkflowNodeProps,
  WorkflowNodeRenderer,
} from "@flowgram.ai/free-layout-editor";
import "@flowgram.ai/free-layout-editor/index.css";
import { type JSX, useEffect, useMemo } from "react";
import type { FlowgramDocument, FlowgramNodeData } from "@/app/renderers/flowgram-document";

export interface FlowgramCanvasProps {
  document: FlowgramDocument;
  /** Called with a task id when its node is clicked. */
  onNodeClick?: (taskId: string) => void;
  /** When true the canvas is editable (drag / connect / add / remove). */
  editable?: boolean;
  /** Fires on every edit with the current document (editable mode only). */
  onChange?: (document: FlowgramDocument) => void;
  className?: string;
}

/**
 * molexp-styled node body. Renders inside flowgram's `WorkflowNodeRenderer`
 * (which owns the ports / drag affordances) but the visible card is pure
 * shadcn/Tailwind. Clicking the card surfaces the task id to the host.
 */
const NodeCard = ({ onNodeClick }: { onNodeClick?: (taskId: string) => void }): JSX.Element => {
  const render = useNodeRender();
  const data = (render.data ?? {}) as Partial<FlowgramNodeData>;
  const taskId = data.taskId ?? render.id;
  return (
    <button
      type="button"
      className="flex min-w-[150px] max-w-[260px] cursor-pointer flex-col rounded-md border border-violet-300 bg-card px-3 py-2 text-left shadow-sm transition-colors hover:border-violet-500 dark:border-violet-700"
      onClick={() => onNodeClick?.(taskId)}
    >
      <span className="truncate text-xs font-semibold text-foreground">{data.title ?? taskId}</span>
      <span className="truncate font-mono text-[10px] text-muted-foreground">
        [{data.taskType ?? data.subtitle ?? ""}]
      </span>
    </button>
  );
};

/**
 * Triggers flowgram's dagre layered auto-layout once the canvas has mounted and
 * measured its node rects, replacing the IR's coarse fallback grid
 * (flowgram-document.ts). Must live INSIDE FreeLayoutEditorProvider so
 * `useAutoLayout` can resolve the layout service the free-layout preset
 * registers. Runs a frame after mount, then once more shortly after in case the
 * first pass beat the node-size ResizeObserver.
 */
const AutoLayoutOnMount = (): null => {
  const autoLayout = useAutoLayout();
  useEffect(() => {
    let active = true;
    const run = async () => {
      try {
        await autoLayout();
      } catch (err) {
        console.error("[flowgram auto-layout]", err);
      }
    };
    const raf = requestAnimationFrame(() => {
      if (active) void run();
    });
    const retry = setTimeout(() => {
      if (active) void run();
    }, 250);
    return () => {
      active = false;
      cancelAnimationFrame(raf);
      clearTimeout(retry);
    };
  }, [autoLayout]);
  return null;
};

export const FlowgramCanvas = ({
  document,
  onNodeClick,
  editable = false,
  onChange,
  className,
}: FlowgramCanvasProps): JSX.Element => {
  const editorProps = useMemo<FreeLayoutProps>(() => {
    return {
      background: true,
      readonly: !editable,
      initialData: document,
      nodeRegistries: [],
      // Generic nodes: flowgram auto-assigns a default input + output port
      // (see free-layout-core) so links connect without a custom registry.
      getNodeDefaultRegistry(type) {
        return { type, meta: {} };
      },
      materials: {
        renderDefaultNode: (props: WorkflowNodeProps) => (
          <WorkflowNodeRenderer node={props.node}>
            <NodeCard onNodeClick={onNodeClick} />
          </WorkflowNodeRenderer>
        ),
      },
      // Editing engines are only needed in editable mode.
      ...(editable
        ? {
            nodeEngine: { enable: true },
            history: { enable: true },
            onContentChange(ctx) {
              onChange?.(ctx.document.toJSON() as unknown as FlowgramDocument);
            },
          }
        : {}),
      onAllLayersRendered(ctx) {
        ctx.document.fitView(false);
      },
    };
  }, [document, onNodeClick, editable, onChange]);

  return (
    <div className={`relative h-full w-full ${className ?? ""}`}>
      <FreeLayoutEditorProvider {...editorProps}>
        <EditorRenderer />
        <AutoLayoutOnMount />
      </FreeLayoutEditorProvider>
    </div>
  );
};
