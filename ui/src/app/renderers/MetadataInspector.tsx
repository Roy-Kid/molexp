import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import type { RendererProps } from "@/app/types";
import { buildMetadataFields } from "@/app/renderers/metadata";

export const MetadataInspector = ({ selection, snapshot }: RendererProps): JSX.Element => {
  const fields = buildMetadataFields(selection, snapshot);

  return (
    <Card className="h-full border-border/60 bg-muted/30">
      <CardHeader className="space-y-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base font-semibold">Inspector</CardTitle>
          <Badge variant="secondary" className="uppercase tracking-wide">
            {selection.objectType}
          </Badge>
        </div>
        <p className="text-xs text-muted-foreground">
          Properties reflect the current selection and are read-only in this view.
        </p>
      </CardHeader>
      <Separator />
      <CardContent className="space-y-3 pt-4">
        {fields.map(field => (
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
