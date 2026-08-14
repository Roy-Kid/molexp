import { DashboardCanvas, OverviewSurface } from "@/app/components/entity";
import { formatScalar } from "@/app/renderers/dashboardData";
import type { ApiAssetResponse, RunSummary } from "@/app/types";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { formatDateTime } from "@/lib/datetime";
// Module-path import (not the barrel) — see KnowledgeBacklinksCard loader note.
import { KnowledgeBacklinksCard } from "@/plugins/knowledge";

interface RunOverviewProps {
  run: RunSummary;
  backend: string;
  duration: string | null;
  attemptCount: number;
  assets: ApiAssetResponse[];
  parameters: [string, unknown][];
  results: [string, unknown][];
}

interface KeyValueTableProps {
  entries: [string, unknown][];
  empty: string;
}

/** shadcn Table for key/value science fields — not card rows. */
const KeyValueTable = ({ entries, empty }: KeyValueTableProps): JSX.Element => {
  if (entries.length === 0) {
    return <p className="py-4 text-label text-muted-foreground">{empty}</p>;
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead className="w-2/5">Key</TableHead>
          <TableHead>Value</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {entries.map(([key, rawValue]) => {
          const value = formatScalar(rawValue);
          return (
            <TableRow key={key}>
              <TableCell className="align-top text-label text-muted-foreground">{key}</TableCell>
              <TableCell className="break-all font-mono text-label text-foreground" title={value}>
                {value}
              </TableCell>
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
};

/**
 * Run overview — single attempt, padded dashboard canvas.
 * Parameters / results use shadcn Table; lineage stays in the inspector.
 */
export const RunOverview = ({
  run,
  backend,
  duration,
  attemptCount,
  assets,
  parameters,
  results,
}: RunOverviewProps): JSX.Element => {
  const ops: string[] = [];
  if (run.startedAt) ops.push(`started ${formatDateTime(run.startedAt)}`);
  if (duration) ops.push(duration);
  if (backend) ops.push(backend);
  if (attemptCount > 1) ops.push(`${attemptCount} attempts`);
  if (assets.length > 0) ops.push(`${assets.length} assets`);

  return (
    <OverviewSurface>
      <DashboardCanvas className="max-w-4xl space-y-10">
        {run.errorMessage && (
          <section
            className="relative rounded-panel border border-status-failed/25 bg-status-failed-soft px-4 py-3"
            aria-label="Run error"
          >
            <p className="text-label font-medium text-status-failed-foreground">Error</p>
            <pre className="mt-1.5 whitespace-pre-wrap break-words font-mono text-label leading-relaxed text-status-failed-foreground">
              {run.errorMessage}
            </pre>
          </section>
        )}

        {ops.length > 0 && (
          <p className="font-mono text-micro tabular-nums text-muted-foreground">
            {ops.join(" · ")}
          </p>
        )}

        {run.summary ? (
          <p className="max-w-2xl text-body leading-relaxed text-muted-foreground">{run.summary}</p>
        ) : null}

        <div className="grid gap-10 lg:grid-cols-2">
          <section className="min-w-0 space-y-3">
            <h3 className="text-body-lg font-medium text-foreground">
              Parameters
              <span className="ml-2 font-mono text-micro font-normal text-muted-foreground">
                {parameters.length}
              </span>
            </h3>
            <KeyValueTable entries={parameters} empty="No parameters" />
          </section>
          <section className="min-w-0 space-y-3">
            <h3 className="text-body-lg font-medium text-foreground">
              Results
              <span className="ml-2 font-mono text-micro font-normal text-muted-foreground">
                {results.length}
              </span>
            </h3>
            <KeyValueTable entries={results} empty="No results" />
          </section>
        </div>

        <KnowledgeBacklinksCard
          kind="run"
          projectId={run.projectId}
          experimentId={run.experimentId}
          runId={run.id}
        />
      </DashboardCanvas>
    </OverviewSurface>
  );
};
