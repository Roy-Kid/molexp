import { PanelRightClose, PanelRightOpen, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface ContextBarProps {
  inspectorOpen: boolean;
  searchQuery: string;
  selectionActive: boolean;
  onSearchChange: (query: string) => void;
  onToggleInspector: () => void;
}

export const ContextBar = ({
  inspectorOpen,
  searchQuery,
  selectionActive,
  onSearchChange,
  onToggleInspector,
}: ContextBarProps): JSX.Element => {
  return (
    <header className="border-b border-border bg-background/95 backdrop-blur">
      <div className="flex items-center justify-between gap-6 px-4 py-2 md:px-6">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-foreground text-xs font-semibold uppercase tracking-[0.2em] text-background">
            M
          </div>
          <span className="text-sm font-semibold tracking-tight text-foreground">molexp</span>
        </div>

        <div className="flex w-full max-w-md items-center gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              className="h-8 pl-9"
              placeholder="Search projects, experiments, runs, assets"
              value={searchQuery}
              onChange={(event) => onSearchChange(event.target.value)}
            />
          </div>
          <Button
            variant="outline"
            size="sm"
            className="h-8 gap-2"
            onClick={onToggleInspector}
            disabled={!selectionActive}
          >
            {inspectorOpen ? (
              <PanelRightClose className="h-4 w-4" />
            ) : (
              <PanelRightOpen className="h-4 w-4" />
            )}
            Details
          </Button>
        </div>
      </div>
    </header>
  );
};
