import { Bell, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

export const TopBar = (): JSX.Element => {
  return (
    <header className="grid h-14 grid-cols-1 items-center border-b border-border bg-background px-4 md:grid-cols-3 md:px-6">
      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-full bg-foreground text-xs font-bold uppercase text-background">
          M
        </div>
        <span className="text-sm font-semibold uppercase tracking-widest text-muted-foreground">
          Molexp
        </span>
      </div>

      <div className="hidden items-center justify-center justify-self-center md:flex">
        <span className="text-sm font-semibold uppercase tracking-widest text-foreground">
          Workflow Studio
        </span>
      </div>

      <div className="flex items-center justify-end justify-self-end gap-3">
        <div className="hidden items-center gap-3 lg:flex">
          <Select defaultValue="workspace">
            <SelectTrigger className="h-9 w-36">
              <SelectValue placeholder="Scope" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="workspace">Workspace</SelectItem>
              <SelectItem value="system">System</SelectItem>
              <SelectItem value="admin">Admin</SelectItem>
            </SelectContent>
          </Select>

          <div className="relative w-64">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              className="h-9 pl-9"
              placeholder="Search projects, workflows, runs, assets"
            />
          </div>
        </div>

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
