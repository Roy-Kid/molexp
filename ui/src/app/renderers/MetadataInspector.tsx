import { buildMetadataFields } from "@/app/renderers/metadata";
import type { RendererProps } from "@/app/types";
import { WorkbenchTag } from "@/components/workbench";

export const MetadataInspector = ({ selection, snapshot }: RendererProps): JSX.Element => {
  const fields = buildMetadataFields(selection, snapshot);

  return (
    <div className="flex h-full flex-col bg-background">
      <div className="flex items-center justify-between border-b border-border/70 bg-muted/20 px-3 py-2">
        <h2 className="text-micro font-medium uppercase tracking-wide text-muted-foreground">
          Details
        </h2>
        <WorkbenchTag className="h-5 px-2 text-micro uppercase tracking-wide">
          {selection.objectType}
        </WorkbenchTag>
      </div>
      <dl className="flex-1 divide-y divide-border/50 overflow-auto">
        {fields.map((field) => (
          <div key={field.label} className="px-3 py-2">
            <dt className="text-micro font-medium uppercase tracking-wide text-muted-foreground">
              {field.label}
            </dt>
            <dd className="mt-1 break-words text-label text-foreground">{field.value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
};
