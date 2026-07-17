import type { ReactNode } from "react";

import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";

interface OverviewPageProps {
  children: ReactNode;
  aside?: ReactNode;
  className?: string;
}

export const OverviewPage = ({ children, aside, className }: OverviewPageProps): JSX.Element => {
  return (
    <div className={cn("flex-1 overflow-auto", className)}>
      <div
        className={cn(
          "grid min-h-full gap-x-8 gap-y-8 p-4 md:p-6",
          aside && "xl:grid-cols-[minmax(0,1fr)_minmax(260px,320px)]",
        )}
      >
        <div className="min-w-0 space-y-8">{children}</div>
        {aside && (
          <aside className="min-w-0 space-y-6 border-t border-border/60 pt-6 xl:border-l xl:border-t-0 xl:pl-8 xl:pt-0">
            {aside}
          </aside>
        )}
      </div>
    </div>
  );
};

interface OverviewSectionProps {
  title: string;
  description?: ReactNode;
  children: ReactNode;
  className?: string;
}

export const OverviewSection = ({
  title,
  description,
  children,
  className,
}: OverviewSectionProps): JSX.Element => {
  return (
    <section className={cn("space-y-3", className)}>
      <div>
        <h3 className="text-sm font-medium text-foreground">{title}</h3>
        {description && (
          <p className="mt-1 max-w-2xl text-sm leading-relaxed text-muted-foreground">
            {description}
          </p>
        )}
      </div>
      <Separator className="opacity-60" />
      {children}
    </section>
  );
};

interface OverviewHighlightProps {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
}

export const OverviewHighlight = ({
  label,
  value,
  detail,
}: OverviewHighlightProps): JSX.Element => {
  return (
    <div className="rounded-lg border border-border/70 bg-card px-3 py-2.5 shadow-none">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-1 min-w-0 break-words text-lg font-semibold leading-6 tracking-tight text-foreground">
        {value}
      </div>
      {detail && <div className="mt-1 text-xs leading-5 text-muted-foreground">{detail}</div>}
    </div>
  );
};

interface OverviewHighlightGridProps {
  children: ReactNode;
}

export const OverviewHighlightGrid = ({ children }: OverviewHighlightGridProps): JSX.Element => {
  return <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1">{children}</div>;
};
