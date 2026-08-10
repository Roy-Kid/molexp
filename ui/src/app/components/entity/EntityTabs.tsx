import type { ComponentProps, ReactNode } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";

export interface EntityTabItem {
  value: string;
  label: ReactNode;
  disabled?: boolean;
}

export const EntityTabs = ({ className, ...props }: ComponentProps<typeof Tabs>): JSX.Element => {
  return (
    <Tabs className={cn("flex flex-1 flex-col gap-0 overflow-hidden", className)} {...props} />
  );
};

interface EntityTabBarProps {
  tabs: EntityTabItem[];
  className?: string;
  listClassName?: string;
  triggerClassName?: string;
}

/**
 * Entity tab strip — MolVis PanelTabStrip topology, text labels (no icons).
 * Tabs share the full band evenly; active state is accent text + hairline underline.
 */
export const EntityTabBar = ({
  tabs,
  className,
  listClassName,
  triggerClassName,
}: EntityTabBarProps): JSX.Element => {
  return (
    <div
      className={cn(
        "flex h-9 w-full min-w-0 shrink-0 items-stretch overflow-hidden border-b border-border/60 bg-surface",
        className,
      )}
    >
      <TabsList
        variant="line"
        className={cn(
          "flex h-full w-full min-w-0 gap-0 overflow-hidden rounded-none border-0 bg-transparent p-0",
          "group-data-[orientation=horizontal]/tabs:h-full",
          listClassName,
        )}
      >
        {tabs.map((tab) => (
          <TabsTrigger
            key={tab.value}
            value={tab.value}
            disabled={tab.disabled}
            className={cn(
              "h-full min-w-0 flex-1 basis-0 self-stretch rounded-none border-0 bg-transparent px-2 shadow-none",
              "text-label font-medium text-muted-foreground hover:text-foreground",
              "focus-visible:border-transparent focus-visible:ring-0 focus-visible:ring-offset-0",
              "focus-visible:bg-interactive/60",
              "data-[state=active]:border-transparent data-[state=active]:bg-transparent data-[state=active]:text-accent data-[state=active]:shadow-none",
              "group-data-[orientation=horizontal]/tabs:after:inset-x-3 group-data-[orientation=horizontal]/tabs:after:bottom-0 group-data-[orientation=horizontal]/tabs:after:h-px",
              "after:bg-accent data-[state=active]:after:opacity-100",
              triggerClassName,
            )}
          >
            <span className="truncate">{tab.label}</span>
          </TabsTrigger>
        ))}
      </TabsList>
    </div>
  );
};

export const EntityTabContent = ({
  className,
  ...props
}: ComponentProps<typeof TabsContent>): JSX.Element => {
  return (
    <TabsContent
      className={cn(
        "m-0 flex flex-1 flex-col overflow-hidden bg-canvas p-0 data-[state=inactive]:hidden",
        className,
      )}
      {...props}
    />
  );
};
