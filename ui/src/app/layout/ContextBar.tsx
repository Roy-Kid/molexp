import { Loader2, Menu, RefreshCw, Search } from "lucide-react";
import { ApprovalsBell } from "@/app/components/ApprovalsBell";
import { Input } from "@/components/ui/input";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { WorkbenchIconAction } from "@/components/workbench";

interface ContextBarProps {
  searchQuery: string;
  onSearchChange: (query: string) => void;
  onRefresh: () => void;
  isRefreshing?: boolean;
  /** When set, a hamburger button (mobile only) opens the navigation drawer. */
  onMenuClick?: () => void;
}

export const ContextBar = ({
  searchQuery,
  onSearchChange,
  onRefresh,
  isRefreshing = false,
  onMenuClick,
}: ContextBarProps): JSX.Element => {
  return (
    <header className="flex h-11 flex-none items-center border-b border-border bg-background">
      <div className="flex h-full w-full items-center justify-between gap-2 px-3 sm:gap-4 sm:px-4">
        <div className="flex min-w-0 items-center gap-2">
          {onMenuClick && (
            <WorkbenchIconAction
              label="Open navigation"
              size="default"
              className="flex-none md:hidden"
              onClick={onMenuClick}
            >
              <Menu className="h-4 w-4" />
            </WorkbenchIconAction>
          )}
          {/* Product identity once, top-left — constitution §7 */}
          <span className="text-title font-semibold tracking-tight text-foreground">MolExp</span>
        </div>

        <div className="flex flex-1 items-center justify-end gap-2">
          <div className="relative w-full max-w-md">
            <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
            <Input
              className="h-8 pl-8 pr-8"
              placeholder="Filter"
              value={searchQuery}
              onChange={(event) => onSearchChange(event.target.value)}
              aria-label="Filter list"
            />
            <kbd className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 rounded border border-border bg-muted px-2 py-1 text-micro font-medium text-muted-foreground">
              ⌘K
            </kbd>
          </div>
          <ApprovalsBell />
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <WorkbenchIconAction
                  label="Refresh"
                  size="default"
                  className="flex-none"
                  onClick={onRefresh}
                  disabled={isRefreshing}
                >
                  {isRefreshing ? (
                    <Loader2 className="mol-motion-progress-spin h-4 w-4" />
                  ) : (
                    <RefreshCw className="h-4 w-4" />
                  )}
                </WorkbenchIconAction>
              </TooltipTrigger>
              <TooltipContent side="bottom">Refresh current view</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      </div>
    </header>
  );
};
