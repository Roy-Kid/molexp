import { buildMetadataFields } from "@/app/renderers/metadata";
import type { RendererProps } from "@/app/types";
import { Badge } from "@/components/ui/badge";

export const MetadataInspector = ({ selection, snapshot }: RendererProps): JSX.Element => {
  const fields = buildMetadataFields(selection, snapshot);

  return (
    <div className="flex h-full flex-col bg-background">
      <div className="flex items-center justify-between border-b border-border/70 bg-muted/20 px-3 py-1.5">
        <h2 className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
          Details
        </h2>
        <Badge variant="secondary" className="h-5 px-1.5 text-[10px] uppercase tracking-wide">
          {selection.objectType}
        </Badge>
      </div>
      <dl className="flex-1 divide-y divide-border/50 overflow-auto">
        {fields.map((field) => (
          <div key={field.label} className="px-3 py-1.5">
            <dt className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              {field.label}
            </dt>
            <dd className="mt-0.5 break-words text-xs text-foreground">{field.value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
};
