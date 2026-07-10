import { GanttChartSquare, LayoutDashboard, Table2 } from "lucide-react";
import type { JSX } from "react";

import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";

export const RUNS_TABS = ["overview", "jobs", "timeline"] as const;
export type RunsTab = (typeof RUNS_TABS)[number];

const TAB_DEFS: Array<{ id: RunsTab; label: string; icon: typeof LayoutDashboard }> = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "jobs", label: "Jobs", icon: Table2 },
  { id: "timeline", label: "Timeline", icon: GanttChartSquare },
];

export const parseRunsTab = (raw: string | null | undefined): RunsTab => {
  if (raw && (RUNS_TABS as readonly string[]).includes(raw)) {
    return raw as RunsTab;
  }
  return "overview";
};

interface RunsTabBarProps {
  value: RunsTab;
  onChange: (next: RunsTab) => void;
  className?: string;
}

export const RunsTabBar = ({ value, onChange, className }: RunsTabBarProps): JSX.Element => (
  <Tabs value={value} onValueChange={(next) => onChange(next as RunsTab)} className={className}>
    <TabsList
      variant="line"
      className="h-10 w-full justify-start gap-4 rounded-none bg-transparent p-0 sm:gap-5 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
    >
      {TAB_DEFS.map(({ id, label, icon: Icon }) => (
        <TabsTrigger
          key={id}
          value={id}
          className={cn(
            "h-10 flex-none gap-1.5 rounded-none border-0 border-b-2 border-transparent px-0 py-0",
            "text-sm font-medium text-muted-foreground shadow-none after:hidden",
            "data-[state=active]:border-foreground data-[state=active]:bg-transparent data-[state=active]:text-foreground data-[state=active]:shadow-none",
          )}
        >
          <Icon className="h-3.5 w-3.5" />
          {label}
        </TabsTrigger>
      ))}
    </TabsList>
  </Tabs>
);
