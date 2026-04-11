import { useState } from "react";
import type { RendererProps } from "@/app/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

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
      <Card>
        <CardHeader>
          <CardTitle>No Views Available</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            No views are configured for this file type.
          </p>
        </CardContent>
      </Card>
    );
  }

  // If only one tab, render it directly without tabs UI
  if (tabs.length === 1) {
    const TabComponent = tabs[0].component;
    return <TabComponent {...rendererProps} />;
  }

  return (
    <Card className="h-full flex flex-col">
      {title && (
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">{title}</CardTitle>
        </CardHeader>
      )}
      <CardContent className="flex-1 flex flex-col p-0">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col">
          <TabsList className="mx-4 mt-4 w-fit">
            {tabs.map((tab) => (
              <TabsTrigger key={tab.id} value={tab.id}>
                {tab.label}
              </TabsTrigger>
            ))}
          </TabsList>

          {tabs.map((tab) => {
            const TabComponent = tab.component;
            return (
              <TabsContent
                key={tab.id}
                value={tab.id}
                className="flex-1 m-0 p-4 data-[state=inactive]:hidden"
              >
                <TabComponent {...rendererProps} />
              </TabsContent>
            );
          })}
        </Tabs>
      </CardContent>
    </Card>
  );
};
