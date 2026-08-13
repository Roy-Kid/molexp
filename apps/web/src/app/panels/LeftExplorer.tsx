/**
 * Unified left-side explorer chrome (VS Code secondary sidebar).
 *
 * Every LeftPanel tab — Projects, Workspace, Runs, Knowledge, Agent, … —
 * should render through this shell so title bar, toolbar, padding, and
 * scroll behavior stay identical. View-specific content goes in `children`
 * (and optional `toolbar` for search/filter rows).
 */
import type { ComponentType, JSX, ReactNode, SVGProps } from "react";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuTrigger,
} from "@/components/ui/context-menu";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { WorkbenchToggleAction } from "@/components/workbench";
import { cn } from "@/lib/utils";

// ── Icon rail ───────────────────────────────────────────────────────────────

export interface LeftIconRailItem {
  id: string;
  label: string;
  icon: ComponentType<SVGProps<SVGSVGElement>>;
}

export interface LeftIconRailProps {
  items: LeftIconRailItem[];
  activeId: string;
  onSelect: (id: string) => void;
  /** Pinned to the bottom of the rail (e.g. Settings). */
  footer?: LeftIconRailItem | null;
}

export const LeftIconRail = ({
  items,
  activeId,
  onSelect,
  footer = null,
}: LeftIconRailProps): JSX.Element => (
  <TooltipProvider>
    <nav
      className="flex w-14 shrink-0 flex-col items-center gap-1 border-r border-border bg-muted/20 py-3"
      aria-label="Explorer views"
    >
      {items.map((item) => {
        const Icon = item.icon;
        return (
          <Tooltip key={item.id}>
            <TooltipTrigger asChild>
              <WorkbenchToggleAction
                label={item.label}
                pressed={activeId === item.id}
                onClick={() => onSelect(item.id)}
              >
                <Icon className="h-4 w-4" />
              </WorkbenchToggleAction>
            </TooltipTrigger>
            <TooltipContent side="right">{item.label}</TooltipContent>
          </Tooltip>
        );
      })}
      {footer ? (
        <div className="mt-auto">
          <Tooltip>
            <TooltipTrigger asChild>
              <WorkbenchToggleAction
                label={footer.label}
                pressed={activeId === footer.id}
                onClick={() => onSelect(footer.id)}
              >
                <footer.icon className="h-4 w-4" />
              </WorkbenchToggleAction>
            </TooltipTrigger>
            <TooltipContent side="right">{footer.label}</TooltipContent>
          </Tooltip>
        </div>
      ) : null}
    </nav>
  </TooltipProvider>
);

// ── Explorer column ─────────────────────────────────────────────────────────

export interface LeftExplorerProps {
  /** Section title (e.g. "Projects", "Knowledge"). */
  title: string;
  /** Icon actions in the title row (refresh, new, …). */
  actions?: ReactNode;
  /**
   * Optional second row under the title — search, filters, facet chips.
   * Keep view-specific controls here so the title bar stays uniform.
   */
  toolbar?: ReactNode;
  children: ReactNode;
  /**
   * VS Code blank-area context menu (right-click empty space in the body).
   * Pass menu items (e.g. `<TreeMenuItems … />`); omit when unused.
   */
  blankMenu?: ReactNode;
  className?: string;
  /** Extra class on the scrollable body content wrapper. */
  bodyClassName?: string;
}

/**
 * Title bar + optional toolbar + scroll body.
 * Pair with {@link LeftIconRail} for the full left panel.
 */
export const LeftExplorer = ({
  title,
  actions,
  toolbar,
  children,
  blankMenu,
  className,
  bodyClassName,
}: LeftExplorerProps): JSX.Element => {
  const column = (
    <div
      className={cn(
        "flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden bg-background outline-none",
        className,
      )}
    >
      <header className="shrink-0 border-b border-border px-3 py-2">
        <div className="flex h-control-compact items-center justify-between gap-2">
          <h2 className="min-w-0 truncate text-label font-semibold uppercase tracking-wide text-muted-foreground">
            {title}
          </h2>
          {actions ? (
            <div className="flex shrink-0 items-center gap-0.5">{actions}</div>
          ) : null}
        </div>
        {toolbar ? <div className="mt-2 space-y-1.5">{toolbar}</div> : null}
      </header>

      <ScrollArea className="min-h-0 flex-1">
        {/*
          min-h-full + spacer-friendly body so blank-area menus (when wrapped)
          and empty states fill the viewport like VS Code explorer.
        */}
        <div className={cn("min-h-full px-2 py-2", bodyClassName)}>{children}</div>
      </ScrollArea>
    </div>
  );

  if (!blankMenu) {
    return column;
  }

  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>{column}</ContextMenuTrigger>
      <ContextMenuContent className="w-52">{blankMenu}</ContextMenuContent>
    </ContextMenu>
  );
};

/** Shared action-cluster class for title-bar icon buttons. */
export const leftExplorerActionsClass = "flex shrink-0 items-center gap-0.5";
