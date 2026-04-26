import type { ReactNode } from "react";
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
          "grid min-h-full gap-x-10 gap-y-8 p-4 md:p-6",
          aside && "xl:grid-cols-[minmax(0,1fr)_minmax(280px,360px)]",
        )}
      >
        <div className="min-w-0 space-y-7">{children}</div>
        {aside && (
          <aside className="min-w-0 space-y-7 border-t border-border/70 pt-6 xl:border-l xl:border-t-0 xl:pl-8 xl:pt-0">
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
    <section
      className={cn("border-t border-border/70 pt-4 first:border-t-0 first:pt-0", className)}
    >
      <div className="mb-3">
        <h3 className="text-[11px] font-semibold uppercase text-muted-foreground">{title}</h3>
        {description && (
          <p className="mt-1 max-w-2xl text-sm leading-5 text-muted-foreground">{description}</p>
        )}
      </div>
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
    <div className="border-l-2 border-foreground/20 pl-3">
      <div className="text-[11px] font-medium uppercase text-muted-foreground">{label}</div>
      <div className="mt-0.5 min-w-0 break-words text-lg font-semibold leading-6 text-foreground">
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
  return <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-1">{children}</div>;
};
