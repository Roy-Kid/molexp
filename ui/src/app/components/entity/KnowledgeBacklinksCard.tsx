import { BookOpen, Link2 } from "lucide-react";
import { type JSX, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import type { EntityBacklinkRow } from "@/api/generated/models/EntityBacklinkRow";
import { KnowledgeService } from "@/api/generated/services/KnowledgeService";
import { DashboardCard } from "@/app/components/entity/Dashboard";
import { WorkbenchAction, WorkbenchOperationState } from "@/components/workbench";

interface KnowledgeBacklinksCardProps {
  kind: "run" | "experiment";
  projectId: string;
  experimentId: string;
  runId?: string;
  className?: string;
}

/**
 * Knowledge documents citing this entity (vision-loop-10) — a thin read over
 * `GET /api/knowledge/entity-backlinks` (itself a `Bundle.backlinks` delegator).
 * Rows navigate to the Knowledge section. The empty state renders (never
 * hidden) so the knowledge↔entity seam stays discoverable.
 */
export const KnowledgeBacklinksCard = ({
  kind,
  projectId,
  experimentId,
  runId,
  className,
}: KnowledgeBacklinksCardProps): JSX.Element => {
  const navigate = useNavigate();
  const requestKey = `${kind}:${projectId}:${experimentId}:${runId ?? ""}`;
  const [rows, setRows] = useState<EntityBacklinkRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [settledRequestKey, setSettledRequestKey] = useState("");
  const [requestVersion, setRequestVersion] = useState(0);

  useEffect(() => {
    void requestVersion;
    let cancelled = false;
    setRows(null);
    setError(null);
    KnowledgeService.entityBacklinksApiKnowledgeEntityBacklinksGet(
      kind,
      projectId,
      experimentId,
      runId ?? null,
    )
      .then((response) => {
        if (!cancelled) setRows(response.backlinks);
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load linked knowledge");
        }
      })
      .finally(() => {
        if (!cancelled) setSettledRequestKey(requestKey);
      });
    return () => {
      cancelled = true;
    };
  }, [kind, projectId, experimentId, runId, requestKey, requestVersion]);

  return (
    <DashboardCard
      title="Knowledge"
      description={kind === "run" ? "Notes that cite this run" : "Notes that cite this experiment"}
      className={className}
    >
      {settledRequestKey !== requestKey || (rows === null && !error) ? (
        <WorkbenchOperationState
          kind="loading"
          density="compact"
          title="Loading linked notes…"
          skeletonRows={2}
        />
      ) : error ? (
        <WorkbenchOperationState
          kind="error"
          density="compact"
          title="Could not load linked notes"
          detail={error}
          action={
            <WorkbenchAction
              kind="secondary"
              size="compact"
              onClick={() => {
                setRows(null);
                setRequestVersion((version) => version + 1);
              }}
            >
              Retry
            </WorkbenchAction>
          }
        />
      ) : rows === null ? (
        <WorkbenchOperationState
          kind="error"
          density="compact"
          title="Linked notes unavailable"
          detail="The request finished without a result."
          action={
            <WorkbenchAction
              kind="secondary"
              size="compact"
              onClick={() => {
                setRows(null);
                setRequestVersion((version) => version + 1);
              }}
            >
              Retry
            </WorkbenchAction>
          }
        />
      ) : rows.length === 0 ? (
        <WorkbenchOperationState
          kind="empty"
          density="compact"
          title="No linked notes yet"
          detail={
            kind === "run"
              ? "No knowledge documents cite this run."
              : "No knowledge documents cite this experiment."
          }
        />
      ) : (
        <ul className="space-y-1">
          {rows.map((row) => (
            <li key={row.path}>
              <button
                type="button"
                onClick={() =>
                  navigate(`/knowledge/${row.path.split("/").map(encodeURIComponent).join("/")}`)
                }
                className="flex w-full items-center gap-2 truncate rounded-md px-2 py-2 text-left text-sm transition-colors hover:bg-muted/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                title={row.path}
              >
                <BookOpen className="h-3.5 w-3.5 flex-none text-muted-foreground" />
                <span className="min-w-0 flex-1 truncate text-foreground">{row.title}</span>
                <span className="inline-flex items-center gap-1 font-mono text-micro text-muted-foreground">
                  <Link2 className="h-3 w-3" />
                  {row.role}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </DashboardCard>
  );
};
