import { type JSX, lazy, Suspense } from "react";
import type { FlowgramCanvasProps } from "./flowgram-canvas-impl";

export type { FlowgramCanvasProps } from "./flowgram-canvas-impl";

const Impl = lazy(() =>
  import("./flowgram-canvas-impl").then((mod) => ({ default: mod.FlowgramCanvas })),
);

export const FlowgramCanvas = (props: FlowgramCanvasProps): JSX.Element => (
  <Suspense
    fallback={
      <div className="h-full w-full bg-canvas" role="status" aria-label="Loading workflow canvas" />
    }
  >
    <Impl {...props} />
  </Suspense>
);
