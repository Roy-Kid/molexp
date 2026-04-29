import { GanttChartSquare, LayoutDashboard, Table2 } from "lucide-react";
import type { JSX } from "react";

import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

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
}

export const RunsTabBar = ({ value, onChange }: RunsTabBarProps): JSX.Element => (
  <Tabs value={value} onValueChange={(next) => onChange(next as RunsTab)}>
    <TabsList variant="line" className="h-8">
      {TAB_DEFS.map(({ id, label, icon: Icon }) => (
        <TabsTrigger key={id} value={id} className="h-7 gap-1.5 px-2.5 text-xs">
          <Icon className="h-3.5 w-3.5" />
          {label}
        </TabsTrigger>
      ))}
    </TabsList>
  </Tabs>
);
