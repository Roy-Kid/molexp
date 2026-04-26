import { useState } from "react";
import { EmptyState, EntityTabBar, EntityTabContent, EntityTabs } from "@/app/components/entity";
import type { RendererProps } from "@/app/types";

interface ViewTab {
  id: string;
  label: string;
  component: (props: RendererProps) => JSX.Element;
}

interface TabbedViewerProps extends RendererProps {
  tabs: ViewTab[];
  defaultTab?: string;
  title?: string;
}

export const TabbedViewer = ({
  tabs,
  defaultTab,
  title,
  ...rendererProps
}: TabbedViewerProps): JSX.Element => {
  const [activeTab, setActiveTab] = useState(defaultTab || tabs[0]?.id || "");

  if (tabs.length === 0) {
    return (
      <div className="flex h-full items-center justify-center bg-background">
        <EmptyState
          title="No views available"
          description="No views are configured for this file type."
        />
      </div>
    );
  }

  // If only one tab, render it directly without tabs UI
  if (tabs.length === 1) {
    const TabComponent = tabs[0].component;
    return <TabComponent {...rendererProps} />;
  }

  return (
    <div className="flex h-full flex-col bg-background">
      {title && (
        <div className="border-b border-border/70 px-4 py-2">
          <h2 className="text-base font-semibold text-foreground">{title}</h2>
        </div>
      )}
      <EntityTabs value={activeTab} onValueChange={setActiveTab}>
        <EntityTabBar tabs={tabs.map((tab) => ({ value: tab.id, label: tab.label }))} />

        {tabs.map((tab) => {
          const TabComponent = tab.component;
          return (
            <EntityTabContent key={tab.id} value={tab.id}>
              <TabComponent {...rendererProps} />
            </EntityTabContent>
          );
        })}
      </EntityTabs>
    </div>
  );
};
