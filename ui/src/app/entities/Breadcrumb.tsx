// Shell-level breadcrumb bar. Driven by ``buildTrail`` from the current
// selection, it lives once in the AppShell instead of being re-derived and
// re-rendered inside every entity viewer.

import { ChevronRight } from "lucide-react";
import { Fragment, type JSX } from "react";
import { Link } from "react-router-dom";
import type { BreadcrumbItem } from "@/app/types";
import { cn } from "@/lib/utils";

interface BreadcrumbProps {
  items: BreadcrumbItem[];
}

export const Breadcrumb = ({ items }: BreadcrumbProps): JSX.Element => {
  return (
    <nav className="flex min-w-0 items-center gap-1 overflow-hidden text-label text-muted-foreground">
      {items.map((item, index) => {
        const isLast = index === items.length - 1;
        return (
          <Fragment key={`${item.label}-${item.to ?? index}`}>
            {index > 0 && <ChevronRight className="h-3 w-3 flex-none opacity-50" />}
            {item.to && !isLast ? (
              <Link
                to={item.to}
                className="min-w-0 truncate rounded-control px-1 py-1 transition-colors hover:bg-muted/60 hover:text-foreground"
              >
                {item.label}
              </Link>
            ) : (
              <span
                title={item.label}
                className={cn("min-w-0 truncate", isLast && "font-medium text-foreground")}
              >
                {item.label}
              </span>
            )}
          </Fragment>
        );
      })}
    </nav>
  );
};
