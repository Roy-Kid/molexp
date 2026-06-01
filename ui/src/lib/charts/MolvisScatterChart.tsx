import type { ScatterChartConfig, ScatterClickEvent } from "@molcrafts/molvis-core/charts";
import type { JSX } from "react";
import { useEffect, useImperativeHandle, useRef } from "react";

export interface MolvisScatterChartHandle {
  /** Highlight a single point by index, or pass null to clear. */
  setHighlight(index: number | null): Promise<void>;
}

interface MolvisScatterChartProps {
  config: ScatterChartConfig;
  onPointClick?: (event: ScatterClickEvent) => void;
  /** Imperative handle for cross-pane coordination (e.g. setHighlight). */
  ref?: React.Ref<MolvisScatterChartHandle>;
  className?: string;
  style?: React.CSSProperties;
}

/** React wrapper around molvis-core's imperative ``ScatterChart``. */
export const MolvisScatterChart = ({
  config,
  onPointClick,
  ref,
  className,
  style,
}: MolvisScatterChartProps): JSX.Element => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const clickRef = useRef(onPointClick);
  const chartRef = useRef<{
    dispose: () => void;
    onPointClick: (cb: (e: ScatterClickEvent) => void) => () => void;
    setHighlight: (idx: number | null) => Promise<void>;
  } | null>(null);
  useEffect(() => {
    clickRef.current = onPointClick;
  }, [onPointClick]);

  useImperativeHandle(
    ref,
    () => ({
      setHighlight: async (index: number | null) => {
        await chartRef.current?.setHighlight(index);
      },
    }),
    [],
  );

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }
    let unsubscribe: (() => void) | null = null;
    let cancelled = false;
    void (async () => {
      const { ScatterChart } = await import("@molcrafts/molvis-core/charts");
      if (cancelled) return;
      chartRef.current = new ScatterChart(container, config);
      unsubscribe = chartRef.current.onPointClick((e) => clickRef.current?.(e));
    })();
    return () => {
      cancelled = true;
      unsubscribe?.();
      chartRef.current?.dispose();
      chartRef.current = null;
    };
  }, [config]);

  return <div ref={containerRef} className={className} style={style} />;
};
