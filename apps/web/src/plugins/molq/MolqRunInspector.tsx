import { ServerCog } from "lucide-react";
import { useMemo } from "react";
import { MetadataInspector } from "@/app/renderers/MetadataInspector";
import type { RendererProps } from "@/app/types";
import { WorkbenchTag } from "@/components/workbench";

const formatExecutorLabel = (key: string): string => {
  return key.replace(/_/g, " ").replace(/\b\w/g, (match) => match.toUpperCase());
};

export const MolqRunInspector = (props: RendererProps): JSX.Element => {
  const run = useMemo(() => {
    return props.snapshot.runs.find((item) => item.id === props.selection.objectId) ?? null;
  }, [props.selection.objectId, props.snapshot.runs]);

  if (run?.executorInfo.backend !== "molq") {
    return <MetadataInspector {...props} />;
  }

  const rows = Object.entries(run.executorInfo);

  return (
    <div className="flex h-full flex-col bg-background">
      <div className="flex items-center justify-between border-b border-border/70 bg-muted/20 px-3 py-2">
        <div className="flex min-w-0 items-center gap-2">
          <ServerCog className="h-3.5 w-3.5 text-muted-foreground" />
          <h2 className="truncate text-micro font-medium uppercase tracking-wide text-muted-foreground">
            Executor
          </h2>
        </div>
        <WorkbenchTag className="h-5 px-2 text-micro uppercase tracking-wide">molq</WorkbenchTag>
      </div>
      <dl className="flex-1 divide-y divide-border/50 overflow-auto">
        {rows.map(([key, value]) => (
          <div key={key} className="px-3 py-2">
            <dt className="text-micro font-medium uppercase tracking-wide text-muted-foreground">
              {formatExecutorLabel(key)}
            </dt>
            <dd className="mt-1 break-words font-mono text-label text-foreground">{value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
};
