import { Bell, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface TopBarProps {
  searchQuery?: string;
  onSearchChange?: (query: string) => void;
}

export const TopBar = ({ searchQuery = "", onSearchChange }: TopBarProps): JSX.Element => {
  return (
    <header className="flex h-14 items-center justify-between border-b border-border bg-background px-4 md:px-6">
      {/* Left: Logo */}
      <div className="flex items-center gap-3 w-[200px]">
        <div className="flex h-9 w-9 items-center justify-center rounded-full bg-foreground text-xs font-bold uppercase text-background">
          M
        </div>
        <span className="text-sm font-semibold uppercase tracking-widest text-muted-foreground">
          Molexp
        </span>
      </div>

      {/* Center: Search */}
      <div className="flex flex-1 items-center justify-center">
        <div className="relative w-full max-w-md">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            className="h-9 pl-9"
            placeholder="Search projects, workflows, runs, assets"
            value={searchQuery}
            onChange={(e) => onSearchChange?.(e.target.value)}
          />
        </div>
      </div>

      {/* Right: Icons */}
      <div className="flex items-center justify-end gap-3 w-[200px]">
        <Button variant="ghost" size="icon" aria-label="Notifications">
          <Bell className="h-5 w-5" />
        </Button>
        <div className="flex h-9 w-9 items-center justify-center rounded-full border border-border bg-muted text-xs font-semibold">
          UA
        </div>
      </div>
    </header>
  );
};
