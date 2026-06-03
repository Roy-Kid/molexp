import { BaseEdge, type EdgeProps } from "@xyflow/react";
import type { JSX } from "react";
import type { ElkPoint } from "@/app/renderers/elkLayout";

/**
 * Renders an edge along ELK's computed orthogonal bend points (passed in
 * ``data.points``) so fan-in edges route around nodes. Falls back to a straight
 * segment if ELK gave no routing for this edge.
 */
const RADIUS = 8;

const buildRoundedPath = (points: ElkPoint[]): string => {
  if (points.length < 2) return "";
  if (points.length === 2) {
    return `M ${points[0].x},${points[0].y} L ${points[1].x},${points[1].y}`;
  }
  let d = `M ${points[0].x},${points[0].y}`;
  for (let i = 1; i < points.length - 1; i++) {
    const prev = points[i - 1];
    const curr = points[i];
    const next = points[i + 1];
    // Trim the corner by RADIUS on each side, then quad-curve through it.
    const v1 = norm(curr.x - prev.x, curr.y - prev.y);
    const v2 = norm(next.x - curr.x, next.y - curr.y);
    const r = Math.min(RADIUS, dist(prev, curr) / 2, dist(curr, next) / 2);
    const p1 = { x: curr.x - v1.x * r, y: curr.y - v1.y * r };
    const p2 = { x: curr.x + v2.x * r, y: curr.y + v2.y * r };
    d += ` L ${p1.x},${p1.y} Q ${curr.x},${curr.y} ${p2.x},${p2.y}`;
  }
  const last = points[points.length - 1];
  d += ` L ${last.x},${last.y}`;
  return d;
};

const dist = (a: ElkPoint, b: ElkPoint): number => Math.hypot(b.x - a.x, b.y - a.y);
const norm = (x: number, y: number): ElkPoint => {
  const m = Math.hypot(x, y) || 1;
  return { x: x / m, y: y / m };
};

export const ElkEdge = ({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  data,
  markerEnd,
  style,
}: EdgeProps): JSX.Element => {
  const elkPath = (data?.points as ElkPoint[] | undefined) ?? [];
  // ELK's full orthogonal route (endpoints already on the node borders since
  // ELK's node sizes match the rendered nodes). Fall back to a straight segment
  // between handles only if ELK gave no routing.
  const points: ElkPoint[] =
    elkPath.length >= 2
      ? elkPath
      : [
          { x: sourceX, y: sourceY },
          { x: targetX, y: targetY },
        ];
  const path = buildRoundedPath(points);
  return <BaseEdge id={id} path={path} markerEnd={markerEnd} style={style} />;
};
