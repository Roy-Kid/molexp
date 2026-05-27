import type { GanttChartConfig, GanttClickEvent } from "@molcrafts/molvis-core/charts";
import type { JSX } from "react";
import { useEffect, useRef } from "react";

interface MolvisGanttChartProps {
  config: GanttChartConfig;
  onTaskClick?: (event: GanttClickEvent) => void;
  className?: string;
  style?: React.CSSProperties;
}

/** React wrapper around molvis-core's imperative ``GanttChart``. */
export const MolvisGanttChart = ({
  config,
  onTaskClick,
  className,
  style,
}: MolvisGanttChartProps): JSX.Element => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const clickRef = useRef(onTaskClick);
  useEffect(() => {
    clickRef.current = onTaskClick;
  }, [onTaskClick]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }
    let chartInstance: {
      dispose: () => void;
      onTaskClick: (cb: (e: GanttClickEvent) => void) => () => void;
    } | null = null;
    let unsubscribe: (() => void) | null = null;
    let cancelled = false;
    void (async () => {
      const { GanttChart } = await import("@molcrafts/molvis-core/charts");
      if (cancelled) return;
      chartInstance = new GanttChart(container, config);
      unsubscribe = chartInstance.onTaskClick((e) => clickRef.current?.(e));
    })();
    return () => {
      cancelled = true;
      unsubscribe?.();
      chartInstance?.dispose();
    };
  }, [config]);

  return <div ref={containerRef} className={className} style={style} />;
};
