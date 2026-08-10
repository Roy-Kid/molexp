import type { JSX } from "react";

import { cn } from "@/lib/utils";

export type SparklineTrend = "up" | "down" | "flat";

export interface SparklineProps {
  data: number[];
  height?: number;
  width?: number;
  trend?: SparklineTrend;
  className?: string;
  ariaLabel?: string;
  strokeClassName?: string;
  fillClassName?: string;
}

const TREND_STROKE: Record<SparklineTrend, string> = {
  up: "stroke-success",
  down: "stroke-destructive",
  flat: "stroke-muted-foreground",
};

const TREND_FILL: Record<SparklineTrend, string> = {
  up: "fill-success/10",
  down: "fill-destructive/10",
  flat: "fill-muted-foreground/10",
};

const inferTrend = (data: number[]): SparklineTrend => {
  if (data.length < 2) return "flat";
  const first = data[0];
  const last = data[data.length - 1];
  if (last > first) return "up";
  if (last < first) return "down";
  return "flat";
};

export const Sparkline = ({
  data,
  height = 28,
  width = 96,
  trend,
  className,
  ariaLabel,
  strokeClassName,
  fillClassName,
}: SparklineProps): JSX.Element => {
  const points = data.length === 0 ? [0] : data;
  const min = Math.min(...points);
  const max = Math.max(...points);
  const range = max - min || 1;
  const stepX = points.length > 1 ? width / (points.length - 1) : 0;
  const padY = 2;
  const innerH = Math.max(1, height - padY * 2);

  const coords = points.map((value, index) => {
    const x = points.length === 1 ? width / 2 : index * stepX;
    const y = padY + innerH - ((value - min) / range) * innerH;
    return [x, y] as const;
  });

  const linePath = coords
    .map(([x, y], idx) => `${idx === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`)
    .join(" ");
  const areaPath = `${linePath} L${width.toFixed(2)},${(height - padY).toFixed(2)} L0,${(height - padY).toFixed(2)} Z`;

  const resolvedTrend = trend ?? inferTrend(points);
  const stroke = strokeClassName ?? TREND_STROKE[resolvedTrend];
  const fill = fillClassName ?? TREND_FILL[resolvedTrend];

  return (
    <svg
      role={ariaLabel ? "img" : "presentation"}
      aria-label={ariaLabel}
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      className={cn("h-full w-full overflow-visible", className)}
    >
      <path d={areaPath} className={cn("stroke-none", fill)} />
      <path
        d={linePath}
        className={cn("fill-none", stroke)}
        strokeWidth={1.25}
        strokeLinejoin="round"
        strokeLinecap="round"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
};
