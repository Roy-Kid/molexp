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
import {
  createContext,
  type CSSProperties,
  type JSX,
  useContext,
  useEffect,
  useMemo,
} from "react";
import type { FlowgramDocument, FlowgramNodeData } from "@/app/renderers/flowgram-document";

/**
 * Node display data keyed by task id. flowgram drops arbitrary node `data` for
 * registry-less generic nodes, so we carry it ourselves and `NodeCard` looks it
 * up by the (reliable) node id via this context.
 */
const NodeDataContext = createContext<Map<string, Partial<FlowgramNodeData>>>(new Map());

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

// UML activity-diagram glyph per graph role.
//   input  → initial node (solid "start" pill — UML's filled disc, labelled)
//   output → activity final (double-ring "bullseye" pill)
//   task   → action (rounded rectangle)
const ROLE_STYLES: Record<"input" | "output" | "task", string> = {
  input:
    "rounded-full border-2 border-emerald-500 bg-emerald-100 text-emerald-950 dark:bg-emerald-900/50 dark:text-emerald-50",
  output:
    "rounded-full border-[3px] border-double border-slate-600 bg-slate-100 text-slate-900 dark:border-slate-300 dark:bg-slate-800 dark:text-slate-50",
  task: "rounded-md border-2 border-violet-300 bg-card text-foreground dark:border-violet-700",
};

// Leading UML glyph for terminal roles (initial ●, final ◉).
const ROLE_GLYPH: Partial<Record<"input" | "output" | "task", string>> = {
  input: "●",
  output: "◉",
};

// Corner-dot colour by execution status (varies once a run executes; the
// workflow template is all `pending`).
const STATUS_DOT: Record<string, string> = {
  pending: "bg-slate-300 dark:bg-slate-600",
  running: "bg-blue-500 animate-pulse",
  completed: "bg-emerald-500",
  success: "bg-emerald-500",
  failed: "bg-red-500",
  error: "bg-red-500",
  skipped: "bg-slate-300 ring-1 ring-slate-400",
};

/**
 * molexp-styled node body, rendered as a UML activity-diagram element. Renders
 * inside flowgram's `WorkflowNodeRenderer` (which owns the ports / drag
 * affordances) but the visible card is pure shadcn/Tailwind:
 *   - role → glyph (initial ● / final ◉ / action rectangle);
 *   - a parallel fan-out body → stacked "∥×N" expansion region (run per element);
 *   - the corner dot → execution status.
 * Clicking surfaces the task id to the host.
 */
const NodeCard = ({ onNodeClick }: { onNodeClick?: (taskId: string) => void }): JSX.Element => {
  const render = useNodeRender();
  const nodeDataById = useContext(NodeDataContext);
  const data = nodeDataById.get(render.id) ?? {};
  const taskId = data.taskId ?? render.id;
  const role = data.role ?? "task";
  const status = data.status ?? "pending";
  const parallel = data.parallel ?? false;
  const dot = STATUS_DOT[status] ?? STATUS_DOT.pending;
  // Fake stacked cards behind the node = UML expansion-region multiplicity.
  const stack = parallel
    ? "shadow-[5px_5px_0_-2px] shadow-violet-300 dark:shadow-violet-800"
    : "shadow-sm";
  return (
    <button
      type="button"
      title={`${taskId} · ${role}${parallel ? " · parallel ×N" : ""} · ${status}`}
      className={`relative flex min-w-[150px] max-w-[260px] cursor-pointer flex-col px-3 py-2 text-left transition-[filter] hover:brightness-95 ${ROLE_STYLES[role]} ${stack}`}
      onClick={() => onNodeClick?.(taskId)}
    >
      <span className={`absolute right-1.5 top-1.5 h-2 w-2 rounded-full ${dot}`} aria-hidden />
      <span className="flex items-center gap-1 truncate pr-3 text-xs font-semibold">
        {ROLE_GLYPH[role] && (
          <span className="shrink-0 text-[10px] leading-none" aria-hidden>
            {ROLE_GLYPH[role]}
          </span>
        )}
        {parallel && (
          <span
            className="shrink-0 rounded bg-violet-600 px-1 text-[9px] font-bold text-white"
            aria-hidden
          >
            ∥×N
          </span>
        )}
        <span className="truncate">{data.title ?? taskId}</span>
      </span>
      <span className="truncate font-mono text-[10px] opacity-70">
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

  // Theme the flowgram line/arrow colours (CSS vars it reads) to the molexp
  // design tokens so edges match light/dark instead of flowgram's stock indigo.
  const lineColorVars = {
    "--g-workflow-line-color-default": "hsl(var(--muted-foreground))",
    "--g-workflow-line-color-flowing": "hsl(var(--primary))",
    "--g-workflow-line-color-hover": "hsl(var(--primary))",
    "--g-workflow-line-color-selected": "hsl(var(--primary))",
  } as CSSProperties;

  // Per-node display data keyed by id — buildFlowgramDocument already computed
  // role/status/parallel on each node, so just index it for NodeCard lookup.
  const nodeDataById = useMemo(
    () => new Map(document.nodes.map((n) => [n.id, n.data])),
    [document],
  );

  return (
    <NodeDataContext.Provider value={nodeDataById}>
      <div className={`relative h-full w-full ${className ?? ""}`} style={lineColorVars}>
        <FreeLayoutEditorProvider {...editorProps}>
          <EditorRenderer />
          <AutoLayoutOnMount />
        </FreeLayoutEditorProvider>
      </div>
    </NodeDataContext.Provider>
  );
};
