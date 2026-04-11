import { buildMetadataFields } from "@/app/renderers/metadata";
import type { RendererProps } from "@/app/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

export const MetadataViewer = ({ selection, snapshot }: RendererProps): JSX.Element => {
  const fields = buildMetadataFields(selection, snapshot);

  return (
    <Card className="h-full border-border/60 bg-background">
      <CardHeader className="space-y-2">
        <CardTitle className="text-lg font-semibold">Overview</CardTitle>
        <p className="text-sm text-muted-foreground">
          Semantic metadata sourced from the workspace backend.
        </p>
      </CardHeader>
      <Separator />
      <CardContent className="space-y-4 pt-4">
        {fields.map((field) => (
          <div key={field.label} className="space-y-1">
            <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              {field.label}
            </p>
            <p className="text-sm font-medium text-foreground">{field.value}</p>
          </div>
        ))}
      </CardContent>
    </Card>
  );
};
