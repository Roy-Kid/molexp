import type { RawChartConfig } from "@molcrafts/molplot";
import type { JSX } from "react";
import { useEffect, useRef } from "react";

interface MolplotRawChartProps {
  spec: RawChartConfig;
  className?: string;
  style?: React.CSSProperties;
}

/**
 * React wrapper around molvis-core's ``RawChart`` — for callers that
 * receive arbitrary plotly specs (e.g. agent-emitted visualizations).
 *
 * Spec updates flow through ``RawChart.update()`` (a cheap
 * ``plotly.react`` diff) rather than tearing down + re-mounting, so an
 * SSE-driven caller that hands a new spec object every render gets
 * incremental redraws and preserves zoom/pan state.
 */
export const MolplotRawChart = ({ spec, className, style }: MolplotRawChartProps): JSX.Element => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<{
    dispose: () => void;
    update: (spec: RawChartConfig) => Promise<void>;
  } | null>(null);
  // Keep the latest spec in a ref so the mount-once effect can hand it
  // to the constructor without listing `spec` as a dep (which would
  // tear down + re-mount on every spec change).
  const specRef = useRef(spec);
  specRef.current = spec;

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    let cancelled = false;
    void (async () => {
      const { RawChart } = await import("@molcrafts/molplot");
      if (cancelled) return;
      chartRef.current = new RawChart(container, specRef.current);
    })();
    return () => {
      cancelled = true;
      chartRef.current?.dispose();
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    // Fire-and-forget update; if the chart hasn't mounted yet, the
    // mount effect above will pick up the latest specRef value.
    void chartRef.current?.update(spec);
  }, [spec]);

  return <div ref={containerRef} className={className} style={style} />;
};
