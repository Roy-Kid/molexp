import { BookOpen, Link2 } from "lucide-react";
import { type JSX, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import type { EntityBacklinkRow } from "@/api/generated/models/EntityBacklinkRow";
import { KnowledgeService } from "@/api/generated/services/KnowledgeService";
import { DashboardCard } from "@/app/components/entity/Dashboard";
import { Skeleton } from "@/components/ui/skeleton";

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
  const [rows, setRows] = useState<EntityBacklinkRow[] | null>(null);

  useEffect(() => {
    let cancelled = false;
    KnowledgeService.entityBacklinksApiKnowledgeEntityBacklinksGet(
      kind,
      projectId,
      experimentId,
      runId ?? null,
    )
      .then((response) => {
        if (!cancelled) setRows(response.backlinks);
      })
      .catch(() => {
        if (!cancelled) setRows([]);
      });
    return () => {
      cancelled = true;
    };
  }, [kind, projectId, experimentId, runId]);

  return (
    <DashboardCard
      title="Knowledge"
      description={kind === "run" ? "Notes that cite this run" : "Notes that cite this experiment"}
      className={className}
    >
      {rows === null ? (
        <div className="space-y-2">
          <Skeleton className="h-7 w-full" />
          <Skeleton className="h-7 w-[80%]" />
        </div>
      ) : rows.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          {kind === "run" ? "Harvest this run to link notes." : "No linked notes yet."}
        </p>
      ) : (
        <ul className="space-y-0.5">
          {rows.map((row) => (
            <li key={row.path}>
              <button
                type="button"
                onClick={() =>
                  navigate(`/knowledge/${row.path.split("/").map(encodeURIComponent).join("/")}`)
                }
                className="flex w-full items-center gap-2 truncate rounded-md px-2 py-1.5 text-left text-sm transition-colors hover:bg-muted/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                title={row.path}
              >
                <BookOpen className="h-3.5 w-3.5 flex-none text-muted-foreground" />
                <span className="min-w-0 flex-1 truncate text-foreground">{row.title}</span>
                <span className="inline-flex items-center gap-1 font-mono text-[10px] text-muted-foreground">
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
