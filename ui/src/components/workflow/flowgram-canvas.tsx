/**
 * FlowgramCanvas — the single read-only workflow canvas, built on the
 * `@flowgram.ai/free-layout-editor` free-layout core.
 *
 * Every workflow surface (the run "what ran" preview, the workflow Graph tab,
 * the workspace `workflow.json` viewer) renders through this one component.
 * Nodes are drawn with molexp's own shadcn/ui + Tailwind chrome (NO FlowGram
 * form-materials / Semi Design / Ant Design). The canvas is read-only by
 * default; pass `editable` to enable drag / connect / add / remove with
 * undo-redo history, and `onChange` to receive the edited document for
 * write-back (see WorkflowGraphViewer).
 */

import {
  EditorRenderer,
  FreeLayoutEditorProvider,
  type FreeLayoutProps,
  useAutoLayout,
  useClientContext,
  useNodeRender,
  usePlaygroundTools,
  type WorkflowNodeProps,
  WorkflowNodeRenderer,
} from "@flowgram.ai/free-layout-editor";
import "@flowgram.ai/free-layout-editor/index.css";
import { Layers, Maximize2, Minus, Plus, Redo2, Undo2 } from "lucide-react";
import { createContext, type JSX, useContext, useEffect, useMemo, useRef, useState } from "react";
import { StatusIcon, type StatusKey, statusKey } from "@/app/components/entity";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import {
  buildFlowgramDocument,
  type FlowgramDocument,
  type FlowgramNodeData,
} from "@/components/workflow/flowgram-document";
import type { TaskGraphJson } from "@/components/workflow/task-graph-ir";

const prefersReducedMotion = (): boolean =>
  typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/**
 * Node display data keyed by task id. flowgram drops arbitrary node `data` for
 * registry-less generic nodes, so we carry it ourselves and `NodeCard` looks it
 * up by the (reliable) node id via this context.
 */
const NodeDataContext = createContext<Map<string, Partial<FlowgramNodeData>>>(new Map());

/**
 * Lets a `SubWorkflow` node ask the canvas to open its inner graph in a
 * read-only drill-down dialog. Undefined outside a `FlowgramCanvas`.
 */
const SubworkflowExpandContext = createContext<
  ((taskId: string, inner: TaskGraphJson) => void) | undefined
>(undefined);

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

// SHAPE encodes the graph ROLE (UML activity-diagram glyph) — only the silhouette
// (border-radius + border treatment), never the colour:
//   input  → initial node (solid "start" pill — UML's filled disc, labelled)
//   output → activity final (double-ring "bullseye" pill)
//   task   → action (rounded rectangle)
const ROLE_SHAPE: Record<"input" | "output" | "task", string> = {
  input: "rounded-full border-2",
  output: "rounded-full border-[3px] border-double",
  task: "rounded-md border-2",
};

// Leading UML glyph for terminal roles (initial ●, final ◉).
const ROLE_GLYPH: Partial<Record<"input" | "output" | "task", string>> = {
  input: "●",
  output: "◉",
};

// COLOUR encodes execution STATUS — border + soft fill (+ a steady glow ring for
// running). Drawn on top of the role shape so type and state read independently.
// (The workflow template is all `pending`; colours diverge once a run executes.)
const STATUS_VISUAL: Record<StatusKey, string> = {
  running:
    "border-info bg-info-soft text-foreground ring-4 ring-info/25 shadow-[0_0_12px_-2px_hsl(var(--info)/0.55)]",
  success: "border-success bg-success-soft text-foreground",
  failed: "border-destructive bg-destructive/12 text-foreground",
  skipped: "border-muted-foreground/40 border-dashed bg-muted/30 text-muted-foreground",
  pending: "border-border bg-muted/40 text-foreground",
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
  const expandSubworkflow = useContext(SubworkflowExpandContext);
  const data = nodeDataById.get(render.id) ?? {};
  const taskId = data.taskId ?? render.id;
  const role = data.role ?? "task";
  const status = data.status ?? "pending";
  const parallel = data.parallel ?? false;
  const subworkflow = data.subworkflow;
  const visual = STATUS_VISUAL[statusKey(status)];
  // Show WHY a node broke: when it failed, surface the recorded error both in
  // the hover tooltip and as a truncated inline line on the (red) card.
  const failed = statusKey(status) === "failed";
  const error = failed ? data.error : undefined;
  // Fake stacked cards behind the node = UML expansion-region multiplicity.
  const stack = parallel ? "shadow-[5px_5px_0_-2px] shadow-violet-300 dark:shadow-violet-800" : "";
  // A subworkflow node also reads as a (UML rake-glyph) composite — render the
  // expand affordance and a ▣ badge so it is visually distinct.
  return (
    <div className="relative">
      <button
        type="button"
        title={`${taskId} · ${role}${parallel ? " · parallel ×N" : ""}${subworkflow ? " · subworkflow" : ""} · ${status}${error ? `\n${error}` : ""}`}
        className={`relative flex min-w-[150px] max-w-[260px] cursor-pointer flex-col px-3 py-2 text-left transition-[filter] hover:brightness-95 ${ROLE_SHAPE[role]} ${visual} ${stack}`}
        onClick={() => onNodeClick?.(taskId)}
      >
        <StatusIcon status={status} className="absolute right-1.5 top-1.5 h-3.5 w-3.5" />
        <span className="flex items-center gap-1 truncate pr-4 text-xs font-semibold">
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
          {subworkflow && (
            <span
              className="shrink-0 rounded bg-sky-600 px-1 text-[9px] font-bold text-white"
              aria-hidden
            >
              ▣ sub
            </span>
          )}
          <span className="truncate">{data.title ?? taskId}</span>
        </span>
        <span className="truncate font-mono text-[10px] opacity-70">
          [{data.taskType ?? data.subtitle ?? ""}]
        </span>
        {error && (
          <span className="mt-1 line-clamp-2 border-t border-destructive/30 pt-1 font-mono text-[10px] leading-tight text-destructive">
            {error}
          </span>
        )}
      </button>
      {subworkflow && expandSubworkflow && (
        <button
          type="button"
          title="Open inner workflow"
          aria-label={`Open inner workflow of ${taskId}`}
          className="absolute -right-2 -bottom-2 flex h-5 w-5 items-center justify-center rounded-full border border-sky-600 bg-card text-sky-600 shadow-sm transition-colors hover:bg-sky-600 hover:text-white"
          onClick={(event) => {
            event.stopPropagation();
            expandSubworkflow(taskId, subworkflow);
          }}
        >
          <Layers className="h-3 w-3" />
        </button>
      )}
    </div>
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
  // `useAutoLayout()` hands back a fresh bound fn every render, so the effect
  // must NOT depend on it — otherwise every re-render (notably a node drag)
  // re-fires auto-layout + fitView and the node snaps back / the view jumps.
  // Capture the latest fn in a ref and run the layout exactly once on mount.
  const autoLayoutRef = useRef(autoLayout);
  autoLayoutRef.current = autoLayout;
  const ranRef = useRef(false);
  useEffect(() => {
    if (ranRef.current) return;
    ranRef.current = true;
    let active = true;
    const run = async () => {
      try {
        await autoLayoutRef.current();
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
  }, []);
  return null;
};

/**
 * Canvas controls overlay — zoom out / level / zoom in / fit, plus undo-redo in
 * editable mode. Rendered INSIDE the provider so it can reach the playground
 * tools and history service. Motion respects `prefers-reduced-motion`.
 */
const FlowgramCanvasControls = ({ editable }: { editable: boolean }): JSX.Element => {
  const tools = usePlaygroundTools();
  const ctx = useClientContext();
  const easing = !prefersReducedMotion();
  const zoomPct = Math.round((tools.zoom ?? 1) * 100);

  return (
    <div className="absolute right-3 bottom-3 z-10 flex items-center gap-0.5 rounded-md border border-border bg-card p-1 shadow-sm">
      {editable && (
        <>
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            aria-label="Undo"
            onClick={() => ctx.history.undo()}
          >
            <Undo2 className="h-3.5 w-3.5" />
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            aria-label="Redo"
            onClick={() => ctx.history.redo()}
          >
            <Redo2 className="h-3.5 w-3.5" />
          </Button>
          <span aria-hidden="true" className="mx-0.5 h-4 w-px bg-border" />
        </>
      )}
      <Button
        type="button"
        variant="ghost"
        size="icon-sm"
        aria-label="Zoom out"
        onClick={() => tools.zoomout(easing)}
      >
        <Minus className="h-3.5 w-3.5" />
      </Button>
      <span
        aria-live="polite"
        className="min-w-[3.5ch] text-center text-[11px] tabular-nums text-muted-foreground"
      >
        {zoomPct}%
      </span>
      <Button
        type="button"
        variant="ghost"
        size="icon-sm"
        aria-label="Zoom in"
        onClick={() => tools.zoomin(easing)}
      >
        <Plus className="h-3.5 w-3.5" />
      </Button>
      <Button
        type="button"
        variant="ghost"
        size="icon-sm"
        aria-label="Fit to view"
        onClick={() => tools.fitView(easing)}
      >
        <Maximize2 className="h-3.5 w-3.5" />
      </Button>
    </div>
  );
};

export const FlowgramCanvas = ({
  document,
  onNodeClick,
  editable = false,
  onChange,
  className,
}: FlowgramCanvasProps): JSX.Element => {
  // Callbacks are reached through refs so they are NOT memo deps: a parent that
  // passes inline handlers (e.g. onNodeClick) would otherwise rebuild
  // `editorProps` on every render — and rebuilding it reloads `initialData` into
  // flowgram, discarding any in-progress node drag. Keep editorProps stable per
  // (document, editable).
  const onNodeClickRef = useRef(onNodeClick);
  onNodeClickRef.current = onNodeClick;
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;

  const editorProps = useMemo<FreeLayoutProps>(() => {
    // Resolve each edge's status so its colour/animation tracks the run. An edge
    // feeds its target, so it takes the target node's status unless the link
    // itself carries a more specific one. `running` target → flowing animation.
    const nodeStatusById = new Map(document.nodes.map((n) => [n.id, n.data.status ?? "pending"]));
    const edgeStatusByKey = new Map<string, StatusKey>();
    for (const edge of document.edges) {
      const explicit = edge.data?.status ?? edge.status;
      const raw =
        explicit && explicit !== "pending" ? explicit : nodeStatusById.get(edge.targetNodeID);
      edgeStatusByKey.set(`${edge.sourceNodeID}->${edge.targetNodeID}`, statusKey(raw));
    }
    const lineStatus = (line: {
      from?: { id?: string };
      to?: { id?: string };
      info?: { from?: string; to?: string };
    }): StatusKey => {
      const from = line.from?.id ?? line.info?.from;
      const to = line.to?.id ?? line.info?.to;
      return edgeStatusByKey.get(`${from}->${to}`) ?? "pending";
    };

    return {
      background: true,
      readonly: !editable,
      initialData: document,
      nodeRegistries: [],
      // Arrow colour follows status (default grey, blue while flowing, red on
      // error); `setLineClassName` adds the per-status class the stylesheet uses
      // to recolour the gradient stops (success green, skipped grey).
      lineColor: {
        hidden: "transparent",
        default: "hsl(var(--muted-foreground))",
        drawing: "hsl(var(--primary))",
        hovered: "hsl(var(--primary))",
        selected: "hsl(var(--primary))",
        error: "hsl(var(--destructive))",
        flowing: "hsl(var(--info))",
      },
      isFlowingLine: (_ctx, line) => lineStatus(line) === "running",
      setLineClassName: (_ctx, line) => `molexp-edge-${lineStatus(line)}`,
      // Generic nodes: flowgram auto-assigns a default input + output port
      // (see free-layout-core) so links connect without a custom registry.
      getNodeDefaultRegistry(type) {
        return { type, meta: {} };
      },
      materials: {
        renderDefaultNode: (props: WorkflowNodeProps) => (
          <WorkflowNodeRenderer node={props.node}>
            <NodeCard onNodeClick={(id) => onNodeClickRef.current?.(id)} />
          </WorkflowNodeRenderer>
        ),
      },
      // Editing engines are only needed in editable mode.
      ...(editable
        ? {
            nodeEngine: { enable: true },
            history: { enable: true },
            onContentChange(ctx) {
              onChangeRef.current?.(ctx.document.toJSON() as unknown as FlowgramDocument);
            },
          }
        : {}),
      onAllLayersRendered(ctx) {
        ctx.document.fitView(false);
      },
    };
  }, [document, editable]);

  // Per-node display data keyed by id — buildFlowgramDocument already computed
  // role/status/parallel on each node, so just index it for NodeCard lookup.
  const nodeDataById = useMemo(
    () => new Map(document.nodes.map((n) => [n.id, n.data])),
    [document],
  );

  // Read-only drill-down: a SubWorkflow node's expand button opens its inner
  // graph in a dialog. The nested canvas is always read-only, regardless of the
  // outer `editable` (you view inner topology, you don't edit it here).
  const [expanded, setExpanded] = useState<{ taskId: string; inner: TaskGraphJson } | null>(null);
  const innerDocument = useMemo(
    () => (expanded ? buildFlowgramDocument(expanded.inner) : null),
    [expanded],
  );

  return (
    <SubworkflowExpandContext.Provider value={(taskId, inner) => setExpanded({ taskId, inner })}>
      <NodeDataContext.Provider value={nodeDataById}>
        <div className={`relative h-full w-full ${className ?? ""}`}>
          <FreeLayoutEditorProvider {...editorProps}>
            <EditorRenderer />
            <AutoLayoutOnMount />
            <FlowgramCanvasControls editable={editable} />
          </FreeLayoutEditorProvider>
        </div>
      </NodeDataContext.Provider>

      <Dialog open={expanded !== null} onOpenChange={(open) => !open && setExpanded(null)}>
        <DialogContent className="flex h-[80vh] max-w-5xl flex-col">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 font-mono text-sm">
              <Layers className="h-4 w-4 text-sky-600" />
              {expanded?.taskId}
              <span className="text-muted-foreground">· inner workflow</span>
              {expanded?.inner.name && (
                <span className="text-muted-foreground">({expanded.inner.name})</span>
              )}
            </DialogTitle>
          </DialogHeader>
          <div className="min-h-0 flex-1">
            {innerDocument && <FlowgramCanvas document={innerDocument} />}
          </div>
        </DialogContent>
      </Dialog>
    </SubworkflowExpandContext.Provider>
  );
};
