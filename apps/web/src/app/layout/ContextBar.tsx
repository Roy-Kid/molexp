import { CloudOff, HardDrive, Loader2, Menu, RefreshCw, Search, Server } from "lucide-react";
import { UserMenu } from "@/app/auth";
import { ApprovalsBell } from "@/app/components/ApprovalsBell";
import type { ServedWorkspaceSummary } from "@/app/types";
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
  /** Active served workspace — shown next to the product mark so the mount is never anonymous. */
  activeWorkspace?: ServedWorkspaceSummary | null;
}

export const ContextBar = ({
  searchQuery,
  onSearchChange,
  onRefresh,
  isRefreshing = false,
  onMenuClick,
  activeWorkspace = null,
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
          {activeWorkspace && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span
                    className="hidden min-w-0 max-w-[14rem] items-center gap-1.5 truncate rounded-control border border-border/70 bg-muted/40 px-2 py-0.5 text-micro text-muted-foreground sm:inline-flex lg:max-w-[22rem]"
                    title={activeWorkspace.label}
                  >
                    {activeWorkspace.unreachable ? (
                      <CloudOff
                        className="h-3 w-3 flex-none text-status-failed-foreground"
                        aria-hidden
                      />
                    ) : activeWorkspace.isRemote ? (
                      <Server
                        className="h-3 w-3 flex-none text-status-warning-foreground"
                        aria-hidden
                      />
                    ) : (
                      <HardDrive className="h-3 w-3 flex-none text-muted-foreground" aria-hidden />
                    )}
                    <span className="min-w-0 truncate font-mono text-foreground/80">
                      {activeWorkspace.label}
                    </span>
                  </span>
                </TooltipTrigger>
                <TooltipContent side="bottom" className="max-w-sm font-mono text-micro">
                  {activeWorkspace.unreachable
                    ? "Unreachable · "
                    : activeWorkspace.isRemote
                      ? "Remote · "
                      : "Local · "}
                  {activeWorkspace.label}
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
        </div>

        <div className="flex flex-1 items-center justify-end gap-2">
          <div className="relative w-full max-w-md">
            <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
            <Input
              className="h-control pl-8 pr-8"
              placeholder="Filter"
              value={searchQuery}
              onChange={(event) => onSearchChange(event.target.value)}
              aria-label="Filter list"
            />
            <kbd className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 rounded-control border border-border bg-muted px-2 py-1 text-micro font-medium text-muted-foreground">
              ⌘K
            </kbd>
          </div>
          <ApprovalsBell />
          <UserMenu />
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
