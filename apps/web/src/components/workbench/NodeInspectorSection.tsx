import { type JSX, type ReactNode, useId } from "react";

import { cn } from "@/lib/utils";

export interface NodeInspectorSectionProps {
  title: string;
  children: ReactNode;
  className?: string;
}

export const NodeInspectorSection = ({
  title,
  children,
  className,
}: NodeInspectorSectionProps): JSX.Element => {
  const titleId = useId();

  return (
    <section aria-labelledby={titleId} className={cn("mb-3 space-y-2", className)}>
      <h3 id={titleId} className="text-label font-medium text-muted-foreground">
        {title}
      </h3>
      <div className="space-y-1 border-t border-border/80 pt-2">{children}</div>
    </section>
  );
};
