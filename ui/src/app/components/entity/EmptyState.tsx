import type { JSX, ReactNode } from "react";

import { cn } from "@/lib/utils";

export type EmptyStateDensity = "default" | "compact" | "inline";

export interface EmptyStateProps {
  title: string;
  description?: string;
  action?: ReactNode;
  icon?: ReactNode;
  density?: EmptyStateDensity;
}

const CONTAINER: Record<EmptyStateDensity, string> = {
  default: "flex flex-col items-center gap-3 py-14 text-center",
  compact: "flex flex-col items-center gap-2 py-8 text-center",
  inline: "px-2 py-1 text-left",
};

export const EmptyState = ({
  title,
  description,
  action,
  icon,
  density = "default",
}: EmptyStateProps): JSX.Element => {
  return (
    <div className={CONTAINER[density]}>
      {icon && density !== "inline" && (
        <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-border/70 bg-card text-muted-foreground/50 shadow-none">
          {icon}
        </div>
      )}
      {icon && density === "inline" && (
        <div className="mb-1 text-muted-foreground/40">{icon}</div>
      )}
      <div className={cn(density === "inline" ? "space-y-0.5" : "max-w-sm space-y-1")}>
        <p
          className={cn(
            "font-medium text-foreground",
            density === "default" ? "text-sm" : "text-xs",
          )}
        >
          {title}
        </p>
        {description && (
          <p
            className={cn(
              "text-muted-foreground",
              density === "default" ? "text-sm leading-relaxed" : "text-xs leading-relaxed",
            )}
          >
            {description}
          </p>
        )}
      </div>
      {action && <div className={cn(density === "default" ? "mt-1" : "mt-0.5")}>{action}</div>}
    </div>
  );
};
