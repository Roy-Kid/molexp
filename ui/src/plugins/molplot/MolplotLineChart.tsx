import type { LineChartConfig, SeriesPoint } from "@molcrafts/molplot";
import type { JSX } from "react";
import { useEffect, useImperativeHandle, useRef } from "react";

export interface MolplotLineChartHandle {
  /** Push a single point onto an existing series (cheap extendTraces). */
  appendPoint(seriesId: string, point: SeriesPoint): Promise<void>;
  /** Batch-push points; preferred for streaming sources. */
  appendPoints(seriesId: string, points: SeriesPoint[]): Promise<void>;
  /** Replace an entire series in one restyle. */
  setSeries(seriesId: string, points: SeriesPoint[]): Promise<void>;
  /** Update the sliding window; null = unbounded. */
  setWindow(maxPoints: number | null): Promise<void>;
  /** Pin or auto-fit an axis range. */
  setAxisRange(axis: "x" | "y", range: [number, number] | "auto"): Promise<void>;
  /** Clear one series, or every series if no id is given. */
  clear(seriesId?: string): Promise<void>;
}

interface MolplotLineChartProps {
  config: LineChartConfig;
  /** Imperative handle for streaming / cross-pane interaction. */
  ref?: React.Ref<MolplotLineChartHandle>;
  /** Tailwind utility classes for the container. */
  className?: string;
  /** Inline style overrides for the container. */
  style?: React.CSSProperties;
}

/**
 * Thin React wrapper around molvis-core's imperative ``LineChart`` —
 * mounts / disposes the underlying chart instance against a div ref
 * and re-renders when the ``config`` reference changes.
 *
 * Callers passing dynamic configs must memoise them; otherwise every
 * parent render tears down the chart. This intentional sharpness lets
 * the wrapper stay free of deep-equality logic.
 *
 * For streaming use cases reach the imperative API through the ref:
 * a 1Hz metric source should call ``handle.appendPoint(id, point)``
 * rather than re-passing a growing ``config.series[].initialPoints``
 * each tick.
 */
export const MolplotLineChart = ({
  config,
  ref,
  className,
  style,
}: MolplotLineChartProps): JSX.Element => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<{
    dispose: () => void;
    appendPoint: (id: string, p: SeriesPoint) => Promise<void>;
    appendPoints: (id: string, p: SeriesPoint[]) => Promise<void>;
    setSeries: (id: string, p: SeriesPoint[]) => Promise<void>;
    setWindow: (n: number | null) => Promise<void>;
    setAxisRange: (axis: "x" | "y", range: [number, number] | "auto") => Promise<void>;
    clear: (id?: string) => Promise<void>;
  } | null>(null);

  useImperativeHandle(
    ref,
    () => ({
      appendPoint: async (id, point) => chartRef.current?.appendPoint(id, point),
      appendPoints: async (id, points) => chartRef.current?.appendPoints(id, points),
      setSeries: async (id, points) => chartRef.current?.setSeries(id, points),
      setWindow: async (n) => chartRef.current?.setWindow(n),
      setAxisRange: async (axis, range) => chartRef.current?.setAxisRange(axis, range),
      clear: async (id) => chartRef.current?.clear(id),
    }),
    [],
  );

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }
    let cancelled = false;
    void (async () => {
      const { LineChart } = await import("@molcrafts/molplot");
      if (cancelled) return;
      chartRef.current = new LineChart(container, config);
    })();
    return () => {
      cancelled = true;
      chartRef.current?.dispose();
      chartRef.current = null;
    };
  }, [config]);

  return <div ref={containerRef} className={className} style={style} />;
};
