import type { BarChartConfig, BarClickEvent } from "@molcrafts/molplot";
import type { JSX } from "react";
import { useEffect, useRef } from "react";

interface MolplotBarChartProps {
  config: BarChartConfig;
  onBarClick?: (event: BarClickEvent) => void;
  className?: string;
  style?: React.CSSProperties;
}

/** React wrapper around molvis-core's imperative ``BarChart``. */
export const MolplotBarChart = ({
  config,
  onBarClick,
  className,
  style,
}: MolplotBarChartProps): JSX.Element => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  // Keep the latest click handler in a ref so an inline arrow in the
  // parent doesn't re-trigger the mount effect on every render and
  // tear down the chart.
  const clickRef = useRef(onBarClick);
  useEffect(() => {
    clickRef.current = onBarClick;
  }, [onBarClick]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }
    let chartInstance: {
      dispose: () => void;
      onBarClick: (cb: (e: BarClickEvent) => void) => () => void;
    } | null = null;
    let unsubscribe: (() => void) | null = null;
    let cancelled = false;
    void (async () => {
      const { BarChart } = await import("@molcrafts/molplot");
      if (cancelled) return;
      chartInstance = new BarChart(container, config);
      unsubscribe = chartInstance.onBarClick((e) => clickRef.current?.(e));
    })();
    return () => {
      cancelled = true;
      unsubscribe?.();
      chartInstance?.dispose();
    };
  }, [config]);

  return <div ref={containerRef} className={className} style={style} />;
};
