import { buildMetadataFields } from "@/app/renderers/metadata";
import type { RendererProps } from "@/app/types";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";

export const MetadataInspector = ({ selection, snapshot }: RendererProps): JSX.Element => {
  const fields = buildMetadataFields(selection, snapshot);

  return (
    <div className="flex h-full flex-col bg-background">
      <div className="space-y-2 px-4 py-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-[0.18em] text-muted-foreground">
            Details
          </h2>
          <Badge variant="secondary" className="uppercase tracking-wide">
            {selection.objectType}
          </Badge>
        </div>
        <p className="text-sm text-muted-foreground">Read-only metadata for the current selection.</p>
      </div>
      <Separator />
      <div className="flex-1 space-y-4 overflow-auto px-4 py-4">
        {fields.map((field) => (
          <div key={field.label} className="space-y-1 border-b border-border/50 pb-3 last:border-b-0">
            <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
              {field.label}
            </p>
            <p className="break-words text-sm text-foreground">{field.value}</p>
          </div>
        ))}
      </div>
    </div>
  );
};
