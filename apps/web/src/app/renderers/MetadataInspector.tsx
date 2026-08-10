import { CopyButton } from "@/app/components/entity";
import { buildMetadataFields } from "@/app/renderers/metadata";
import type { RendererProps } from "@/app/types";
import { WorkbenchTag } from "@/components/workbench";

/**
 * Scalar details for the right inspector (ids, status, timestamps, hashes).
 * Hierarchical jumps live in RelatedPanel under **Lineage** — not here.
 */
export const MetadataInspector = ({ selection, snapshot }: RendererProps): JSX.Element => {
  const fields = buildMetadataFields(selection, snapshot);

  return (
    <div className="flex flex-col bg-background">
      <div className="flex h-control-compact items-center justify-between border-b border-border px-3">
        <h2 className="text-micro font-medium uppercase tracking-wide text-muted-foreground">
          Details
        </h2>
        <WorkbenchTag className="h-5 px-2 text-micro uppercase tracking-wide">
          {selection.objectType}
        </WorkbenchTag>
      </div>
      <dl className="divide-y divide-border/50">
        {fields.map((field) => {
          const copyable =
            field.label.toLowerCase().includes("id") ||
            field.label.toLowerCase().includes("hash") ||
            field.label === "Config Hash";
          return (
            <div
              key={field.label}
              className="grid grid-cols-(--inspector-grid-columns) items-start gap-2 px-3 py-1.5"
            >
              <dt className="text-micro text-muted-foreground">{field.label}</dt>
              <dd className="flex min-w-0 items-start justify-end gap-0.5 break-words text-right font-mono text-label text-foreground">
                <span className="min-w-0">{field.value || "—"}</span>
                {copyable && field.value ? (
                  <CopyButton value={field.value} label={field.label} className="size-5 shrink-0" />
                ) : null}
              </dd>
            </div>
          );
        })}
      </dl>
    </div>
  );
};
