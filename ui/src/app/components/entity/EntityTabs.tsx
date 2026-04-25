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

export const EntityTabBar = ({
  tabs,
  className,
  listClassName,
  triggerClassName,
}: EntityTabBarProps): JSX.Element => {
  return (
    <div className={cn("shrink-0 border-b border-border/70 bg-background px-4 md:px-6", className)}>
      <TabsList
        variant="line"
        className={cn(
          "h-11 w-full justify-start gap-6 overflow-hidden rounded-none bg-transparent p-0",
          listClassName,
        )}
      >
        {tabs.map((tab) => (
          <TabsTrigger
            key={tab.value}
            value={tab.value}
            disabled={tab.disabled}
            className={cn(
              "h-11 flex-none rounded-none border-0 border-b-2 border-transparent px-0 py-0 text-sm font-medium text-muted-foreground shadow-none after:hidden data-[state=active]:border-foreground data-[state=active]:bg-transparent data-[state=active]:text-foreground data-[state=active]:shadow-none",
              triggerClassName,
            )}
          >
            {tab.label}
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
        "m-0 flex flex-1 flex-col overflow-hidden p-0 data-[state=inactive]:hidden",
        className,
      )}
      {...props}
    />
  );
};
