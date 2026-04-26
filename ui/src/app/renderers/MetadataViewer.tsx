import {
  KeyValueGrid,
  OverviewHighlight,
  OverviewHighlightGrid,
  OverviewPage,
  OverviewSection,
} from "@/app/components/entity";
import { buildMetadataFields } from "@/app/renderers/metadata";
import type { RendererProps } from "@/app/types";

export const MetadataViewer = ({ selection, snapshot }: RendererProps): JSX.Element => {
  const fields = buildMetadataFields(selection, snapshot);
  const status = fields.find((field) => field.label === "Status")?.value;

  return (
    <div className="flex h-full flex-col bg-background">
      <OverviewPage
        aside={
          <OverviewSection title="Highlights">
            <OverviewHighlightGrid>
              <OverviewHighlight label="Type" value={selection.objectType} />
              <OverviewHighlight label="Object ID" value={selection.objectId} />
              {status && <OverviewHighlight label="Status" value={status} />}
            </OverviewHighlightGrid>
          </OverviewSection>
        }
      >
        <OverviewSection
          title="Overview"
          description="Semantic metadata sourced from the workspace backend."
        >
          <KeyValueGrid
            items={fields.map((field) => ({
              label: field.label,
              value: field.value,
            }))}
          />
        </OverviewSection>
      </OverviewPage>
    </div>
  );
};
