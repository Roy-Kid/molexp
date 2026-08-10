/**
 * Molq entity tab — padded dashboard canvas with executor metadata as shadcn Table.
 * Registered as a run tab contribution (value ``molq``); only matched for molq backends.
 */

import { DashboardCanvas, OverviewSurface } from "@/app/components/entity";
import type { RendererProps } from "@/app/types";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

const formatExecutorLabel = (key: string): string =>
  key.replace(/_/g, " ").replace(/\b\w/g, (match) => match.toUpperCase());

export const MolqRunTab = ({ selection, snapshot }: RendererProps): JSX.Element => {
  const run = snapshot.runs.find((item) => item.id === selection.objectId) ?? null;
  const entries = Object.entries(run?.executorInfo ?? {}).sort(([a], [b]) => a.localeCompare(b));

  return (
    <OverviewSurface>
      <DashboardCanvas className="max-w-4xl space-y-6">
        <section className="space-y-3">
          <h3 className="text-body-lg font-medium text-foreground">Molq</h3>
          <p className="text-label text-muted-foreground">
            Submission and cluster fields from the run executor metadata.
          </p>
          {entries.length === 0 ? (
            <p className="py-4 text-label text-muted-foreground">No executor metadata.</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-48">Field</TableHead>
                  <TableHead>Value</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {entries.map(([key, value]) => (
                  <TableRow key={key}>
                    <TableCell className="align-top text-label text-muted-foreground">
                      {formatExecutorLabel(key)}
                    </TableCell>
                    <TableCell className="break-all font-mono text-label text-foreground">
                      {value}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </section>
      </DashboardCanvas>
    </OverviewSurface>
  );
};
