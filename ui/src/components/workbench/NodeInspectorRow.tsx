import type { JSX, ReactNode } from "react";

import { cn } from "@/lib/utils";

export interface NodeInspectorRowProps {
  label: string;
  value: ReactNode;
  mono?: boolean;
}

export const NodeInspectorRow = ({ label, value, mono }: NodeInspectorRowProps): JSX.Element => (
  <div className="grid grid-cols-(--inspector-grid-columns) gap-2 text-body">
    <span className="text-label text-muted-foreground">{label}</span>
    <span
      className={cn(
        "min-w-0 break-words text-right text-foreground",
        mono && "font-mono tabular-nums",
      )}
    >
      {value}
    </span>
  </div>
);
