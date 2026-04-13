import { ServerCog } from "lucide-react";
import { useMemo } from "react";
import { MetadataInspector } from "@/app/renderers/MetadataInspector";
import type { RendererProps } from "@/app/types";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export const MolqRunInspector = (props: RendererProps): JSX.Element => {
  const run = useMemo(() => {
    return props.snapshot.runs.find((item) => item.id === props.selection.objectId) ?? null;
  }, [props.selection.objectId, props.snapshot.runs]);

  if (!run || run.executorInfo.backend !== "molq") {
    return <MetadataInspector {...props} />;
  }

  const rows = Object.entries(run.executorInfo);

  return (
    <Card className="h-full border-cyan-500/20 bg-cyan-50/40">
      <CardHeader className="space-y-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-base font-semibold">
            <ServerCog className="h-4 w-4 text-cyan-700" />
            Molq Inspector
          </CardTitle>
          <Badge className="bg-cyan-600 text-white hover:bg-cyan-700">scheduler</Badge>
        </div>
        <p className="text-xs text-muted-foreground">
          Normalized executor metadata emitted by the molq backend.
        </p>
      </CardHeader>
      <CardContent className="space-y-3 pt-1">
        {rows.map(([key, value]) => (
          <div key={key} className="space-y-1">
            <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              {key}
            </p>
            <p className="font-mono text-xs text-foreground">{value}</p>
          </div>
        ))}
      </CardContent>
    </Card>
  );
};
