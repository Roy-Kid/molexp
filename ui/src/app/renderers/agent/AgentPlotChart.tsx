/**
 * Chat/agent molplot host — interactive zoom + pan without ScrollArea fighting.
 *
 * molplot axis zoom activates in axis gutters (or Shift+wheel). Parent
 * ScrollAreas must not consume those wheels — use a non-passive native wheel
 * listener (React synthetic wheel is often passive, so preventDefault is a no-op).
 * Empty ``params: []`` from agents would block molplot from injecting scale
 * binds — strip them.
 */

import type { VegaLiteSpec } from "@molcrafts/molplot";
import { type JSX, useEffect, useMemo, useRef } from "react";
import { MolplotRawChart } from "@/plugins/molplot";

const prepareSpec = (raw: Record<string, unknown> | VegaLiteSpec): VegaLiteSpec => {
  const spec = { ...(raw as Record<string, unknown>) };
  // RawChart only injects interaction params when `params` is undefined.
  if (Array.isArray(spec.params) && spec.params.length === 0) {
    delete spec.params;
  }
  return spec as VegaLiteSpec;
};

/** Trap wheel so conversation ScrollArea does not scroll while zooming axes. */
const useTrapWheel = (ref: React.RefObject<HTMLElement | null>): void => {
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    // preventDefault only — stopPropagation in capture would block molplot.
    const onWheel = (e: WheelEvent): void => {
      e.preventDefault();
    };
    el.addEventListener("wheel", onWheel, { capture: true, passive: false });
    return () => el.removeEventListener("wheel", onWheel, { capture: true });
  }, [ref]);
};

export const AgentPlotChart = ({
  spec,
  title,
  height = 360,
}: {
  spec: Record<string, unknown> | VegaLiteSpec;
  title?: string;
  height?: number;
}): JSX.Element => {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const prepared = useMemo(() => prepareSpec(spec), [spec]);
  useTrapWheel(hostRef);

  return (
    <div className="space-y-2 rounded-md border border-border/60 bg-card p-3">
      {title ? <p className="text-xs font-medium text-foreground">{title}</p> : null}
      <div
        ref={hostRef}
        className="relative w-full touch-none select-none"
        style={{ height, minHeight: height }}
        onPointerDown={(e) => {
          e.stopPropagation();
        }}
      >
        <MolplotRawChart
          spec={{ spec: prepared, interactive: true, aspectRatio: 16 / 10 }}
          className="h-full w-full"
          style={{ width: "100%", height: "100%" }}
        />
      </div>
      <p className="text-micro text-muted-foreground">
        Scroll on an axis to zoom · drag the plot to pan · Shift+scroll zooms both
      </p>
    </div>
  );
};
