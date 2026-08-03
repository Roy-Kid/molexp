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
import {
  createContext,
  type JSX,
  type MutableRefObject,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { type StatusKey, statusKey } from "@/app/components/entity";
import { canonicalStatusFor } from "@/app/components/entity/status";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { WorkbenchIconAction, WorkflowNode } from "@/components/workbench";
import {
  buildFlowgramDocument,
  type FlowgramDocument,
  type FlowgramNodeData,
} from "@/components/workflow/flowgram-document";
import type { TaskGraphJson } from "@/components/workflow/task-graph-ir";
import { useReducedMotion } from "@/hooks/use-reduced-motion";

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

/**
 * Canvas node body — shared {@link WorkflowNode} chrome inside flowgram's
 * `WorkflowNodeRenderer` (ports / drag stay with flowgram).
 */
const NodeCard = ({ onNodeClick }: { onNodeClick?: (taskId: string) => void }): JSX.Element => {
  const render = useNodeRender();
  const nodeDataById = useContext(NodeDataContext);
  const expandSubworkflow = useContext(SubworkflowExpandContext);
  const data = nodeDataById.get(render.id) ?? {};
  const taskId = data.taskId ?? render.id;
  const role = (data.role ?? "task") as "input" | "output" | "task";
  const status = data.status ?? "pending";
  const parallel = data.parallel ?? false;
  const subworkflow = data.subworkflow;
  const failed = statusKey(status) === "failed";
  const error = failed ? data.error : undefined;
  const displayStatus = canonicalStatusFor(status) ?? "ready";
  const titleTip = `${taskId} · ${role}${parallel ? " · parallel ×N" : ""}${subworkflow ? " · subworkflow" : ""} · ${displayStatus}${error ? `\n${error}` : ""}`;

  return (
    <div className="relative" title={titleTip}>
      <WorkflowNode
        title={data.title ?? taskId}
        taskType={data.taskType ?? data.subtitle ?? null}
        status={status}
        canvasRole={role}
        parallel={parallel}
        subworkflow={Boolean(subworkflow)}
        error={error}
        density="compact"
        onActivate={onNodeClick ? () => onNodeClick(taskId) : undefined}
      />
      {subworkflow && expandSubworkflow && (
        <WorkbenchIconAction
          label={`Open inner workflow of ${taskId}`}
          className="absolute -right-2 -bottom-2 size-5 rounded-full border border-border bg-card"
          onClick={(event) => {
            event.stopPropagation();
            expandSubworkflow(taskId, subworkflow);
          }}
        >
          <Layers className="h-3 w-3" />
        </WorkbenchIconAction>
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
const AutoLayoutOnMount = ({ settledRef }: { settledRef: MutableRefObject<boolean> }): null => {
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
      if (active) {
        void run().finally(() => {
          // The mount-time layout passes are done — content changes from here
          // on are USER edits (the gate `onContentChange` checks, so the
          // auto-layout itself never marks the document dirty).
          settledRef.current = true;
        });
      }
    }, 250);
    return () => {
      active = false;
      // An unmounted canvas never settles — but it also never edits.
      cancelAnimationFrame(raf);
      clearTimeout(retry);
    };
  }, [settledRef]);
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
  const reducedMotion = useReducedMotion();
  const easing = !reducedMotion;
  const zoomPct = Math.round((tools.zoom ?? 1) * 100);

  return (
    <div className="absolute right-3 bottom-3 z-10 flex items-center gap-1 rounded-[var(--radius-control)] border border-border bg-surface p-1 mol-motion-fade">
      {editable && (
        <>
          <WorkbenchIconAction label="Undo" onClick={() => ctx.history.undo()}>
            <Undo2 className="h-3.5 w-3.5" />
          </WorkbenchIconAction>
          <WorkbenchIconAction label="Redo" onClick={() => ctx.history.redo()}>
            <Redo2 className="h-3.5 w-3.5" />
          </WorkbenchIconAction>
          <span aria-hidden="true" className="mx-1 h-4 w-px bg-border" />
        </>
      )}
      <WorkbenchIconAction label="Zoom out" onClick={() => tools.zoomout(easing)}>
        <Minus className="h-3.5 w-3.5" />
      </WorkbenchIconAction>
      <span
        aria-live="polite"
        className="min-w-[3.5ch] text-center text-micro tabular-nums text-muted-foreground"
      >
        {zoomPct}%
      </span>
      <WorkbenchIconAction label="Zoom in" onClick={() => tools.zoomin(easing)}>
        <Plus className="h-3.5 w-3.5" />
      </WorkbenchIconAction>
      <WorkbenchIconAction label="Fit to view" onClick={() => tools.fitView(easing)}>
        <Maximize2 className="h-3.5 w-3.5" />
      </WorkbenchIconAction>
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
  // False until the mount-time auto-layout passes finish (AutoLayoutOnMount).
  const layoutSettledRef = useRef(false);

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
        default: "var(--molexp-muted-foreground)",
        drawing: "var(--molexp-accent)",
        hovered: "var(--molexp-accent)",
        selected: "var(--molexp-accent)",
        error: "var(--status-failed)",
        flowing: "var(--status-running)",
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
              // The mount-time auto-layout also mutates the document; only
              // post-settle changes are user edits worth a dirty flag (an
              // unguarded page used to warn on unload without any edit).
              if (!layoutSettledRef.current) return;
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
            <AutoLayoutOnMount settledRef={layoutSettledRef} />
            <FlowgramCanvasControls editable={editable} />
          </FreeLayoutEditorProvider>
        </div>
      </NodeDataContext.Provider>

      <Dialog open={expanded !== null} onOpenChange={(open) => !open && setExpanded(null)}>
        <DialogContent className="flex h-[80vh] max-w-5xl flex-col">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 font-mono text-sm">
              <Layers className="h-4 w-4 text-muted-foreground" />
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
