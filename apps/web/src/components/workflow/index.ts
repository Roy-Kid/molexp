/**
 * @molexp workflow components — a self-contained, shadcn-style module for
 * rendering the canonical task-graph IR.
 *
 * Every export here is decoupled from the app shell (no store / router / API):
 * it depends only on React, `@flowgram.ai/free-layout-editor`, the shared
 * shadcn-ui primitives under `@/components/ui`, and props. That makes this the
 * single reuse boundary shared by the in-app workflow viewers and the molexp
 * VSCode extension webview.
 *
 * - {@link WorkflowPreview} — the composable graph + node-details surface.
 * - {@link WorkflowGraph} — the read-only canvas alone.
 * - {@link WorkflowNodeDetails} — the read-only node-detail card.
 * - {@link FlowgramCanvas} / {@link FlowgramCanvasToolbar} — lower-level canvas.
 * - IR types + pure graph utilities + the flowgram-document builder.
 */

export {
  FlowgramCanvas,
  type FlowgramCanvasProps,
} from "@/components/workflow/flowgram-canvas";
export {
  FlowgramCanvasToolbar,
  type FlowgramCanvasToolbarProps,
} from "@/components/workflow/flowgram-canvas-toolbar";
export {
  buildFlowgramDocument,
  type FlowgramDocument,
  type FlowgramEdge,
  type FlowgramNode,
  type FlowgramNodeData,
  flowgramDocToTaskGraphJson,
  normalizeTaskGraph,
  parseTaskGraphIr,
  taskGraphToWireDocument,
} from "@/components/workflow/flowgram-document";
export {
  analyzePaths,
  type GraphEdgeInput,
  type GraphNodeInput,
  planExecution,
  toTaskGraphJson,
} from "@/components/workflow/graph-utils";
export type {
  EdgeJson,
  TaskGraphJson,
  TaskNodeJson,
  TaskNodePosition,
} from "@/components/workflow/task-graph-ir";
export {
  parseWorkflowIr,
  WorkflowGraph,
  type WorkflowIR,
} from "@/components/workflow/workflow-graph";
export {
  WorkflowNodeDetails,
  type WorkflowNodeDetailsProps,
} from "@/components/workflow/workflow-node-details";
export {
  WorkflowPreview,
  type WorkflowPreviewProps,
} from "@/components/workflow/workflow-preview";
