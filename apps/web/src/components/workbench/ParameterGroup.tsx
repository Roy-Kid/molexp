import type { JSX, ReactNode } from "react";

import { cn } from "@/lib/utils";

export interface ParameterGroupProps {
  title: string;
  children: ReactNode;
  className?: string;
}

export const ParameterGroup = ({
  title,
  children,
  className,
}: ParameterGroupProps): JSX.Element => (
  <section className={cn("space-y-2", className)} aria-label={title}>
    <h3 className="text-label font-medium uppercase tracking-wide text-muted-foreground">
      {title}
    </h3>
    <div className="space-y-3 border-t border-border pt-2">{children}</div>
  </section>
);
