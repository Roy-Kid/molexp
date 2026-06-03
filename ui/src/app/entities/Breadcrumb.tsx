// Shell-level breadcrumb bar. Driven by ``buildTrail`` from the current
// selection, it lives once in the AppShell instead of being re-derived and
// re-rendered inside every entity viewer.

import { ChevronRight } from "lucide-react";
import type { JSX } from "react";
import { Link } from "react-router-dom";
import type { BreadcrumbItem } from "@/app/types";

interface BreadcrumbProps {
  items: BreadcrumbItem[];
}

export const Breadcrumb = ({ items }: BreadcrumbProps): JSX.Element => {
  return (
    <nav className="flex min-w-0 flex-wrap items-center gap-1 text-xs text-muted-foreground">
      {items.map((item, index) => {
        const isLast = index === items.length - 1;
        return (
          <span
            key={`${item.label}-${item.to ?? index}`}
            className="flex min-w-0 items-center gap-1"
          >
            {item.to && !isLast ? (
              <Link
                to={item.to}
                className="truncate rounded-sm px-1 py-0.5 transition-colors hover:bg-muted/60 hover:text-foreground"
              >
                {item.label}
              </Link>
            ) : (
              <span className={isLast ? "truncate font-medium text-foreground" : "truncate"}>
                {item.label}
              </span>
            )}
            {!isLast && <ChevronRight className="h-3 w-3 flex-none opacity-50" />}
          </span>
        );
      })}
    </nav>
  );
};
