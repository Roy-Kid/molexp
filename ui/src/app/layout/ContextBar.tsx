import { Loader2, Menu, RefreshCw, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

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
    <header className="border-b border-border bg-background/95 backdrop-blur">
      <div className="flex items-center justify-between gap-2 px-3 py-2 sm:gap-6 sm:px-4 md:px-6">
        <div className="flex items-center gap-2">
          {onMenuClick && (
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 flex-none md:hidden"
              onClick={onMenuClick}
              aria-label="Open navigation"
            >
              <Menu className="h-4 w-4" />
            </Button>
          )}
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-foreground text-xs font-semibold uppercase tracking-[0.2em] text-background">
            M
          </div>
          <span className="hidden text-sm font-semibold tracking-tight text-foreground sm:inline">
            molexp
          </span>
        </div>

        <div className="flex flex-1 items-center justify-end gap-2">
          <div className="relative w-full max-w-md">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              className="h-8 pl-9 pr-16"
              placeholder="Filter current list"
              value={searchQuery}
              onChange={(event) => onSearchChange(event.target.value)}
            />
            {/* The box only filters the visible list. Cross-workspace jump is
                ⌘K (GlobalCommandPalette), hinted here so the scopes are clear. */}
            <kbd className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 rounded border border-border bg-muted px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground">
              ⌘K
            </kbd>
          </div>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 flex-none"
                  onClick={onRefresh}
                  disabled={isRefreshing}
                  aria-label="Refresh"
                >
                  {isRefreshing ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <RefreshCw className="h-4 w-4" />
                  )}
                </Button>
              </TooltipTrigger>
              <TooltipContent side="bottom">Refresh current view</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      </div>
    </header>
  );
};
