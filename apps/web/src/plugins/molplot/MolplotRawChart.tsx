import type { RawChartConfig } from "@molcrafts/molplot";
import type { JSX } from "react";
import { useEffect, useRef } from "react";

interface MolplotRawChartProps {
  spec: RawChartConfig;
  className?: string;
  style?: React.CSSProperties;
}

/**
 * React wrapper around molplot's ``RawChart`` — for callers that receive
 * arbitrary Vega-Lite specs (e.g. agent-emitted visualizations).
 *
 * Spec updates flow through ``RawChart.update()`` only when the serialized
 * payload changes — parent re-renders with a new object identity must not
 * re-embed (that wipes pan/zoom).
 */
export const MolplotRawChart = ({ spec, className, style }: MolplotRawChartProps): JSX.Element => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<{
    dispose: () => void;
    update: (spec: RawChartConfig) => Promise<void>;
  } | null>(null);
  const specRef = useRef(spec);
  const lastJsonRef = useRef<string>("");

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    let cancelled = false;
    void (async () => {
      const { RawChart } = await import("@molcrafts/molplot");
      if (cancelled) return;
      const initial = specRef.current;
      lastJsonRef.current = JSON.stringify(initial);
      chartRef.current = new RawChart(container, {
        ...initial,
        interactive: initial.interactive !== false,
      });
    })();
    return () => {
      cancelled = true;
      chartRef.current?.dispose();
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    const nextJson = JSON.stringify(spec);
    if (nextJson === lastJsonRef.current) return;
    lastJsonRef.current = nextJson;
    specRef.current = spec;
    void chartRef.current?.update({
      ...spec,
      interactive: spec.interactive !== false,
    });
  }, [spec]);

  // Non-passive native wheel trap: React onWheel is often passive under
  // ScrollArea, so preventDefault would be ignored and the chat would scroll
  // instead of molplot axis-zoom.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    // preventDefault only — do not stopPropagation (capture stop would block
    // molplot/vega listeners on descendants from seeing the wheel).
    const onWheel = (e: WheelEvent): void => {
      e.preventDefault();
    };
    el.addEventListener("wheel", onWheel, { capture: true, passive: false });
    return () => el.removeEventListener("wheel", onWheel, { capture: true });
  }, []);

  return <div ref={containerRef} className={className} style={style} />;
};
