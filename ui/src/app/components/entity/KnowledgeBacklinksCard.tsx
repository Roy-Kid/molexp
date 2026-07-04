import { BookOpen, Link2 } from "lucide-react";
import { type JSX, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import type { EntityBacklinkRow } from "@/api/generated/models/EntityBacklinkRow";
import { KnowledgeService } from "@/api/generated/services/KnowledgeService";
import { DashboardCard } from "@/app/components/entity/Dashboard";

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

  const noun = kind === "run" ? "run" : "experiment";
  return (
    <DashboardCard title="Knowledge" className={className}>
      {rows === null ? (
        <p className="text-xs text-muted-foreground">Loading…</p>
      ) : rows.length === 0 ? (
        <p className="text-xs italic text-muted-foreground">
          No knowledge documents cite this {noun} yet.
        </p>
      ) : (
        <ul className="space-y-1">
          {rows.map((row) => (
            <li key={row.path}>
              <button
                type="button"
                onClick={() =>
                  navigate(`/knowledge/${row.path.split("/").map(encodeURIComponent).join("/")}`)
                }
                className="flex w-full items-center gap-2 truncate rounded-sm px-2 py-1 text-left text-sm text-info transition-colors hover:bg-muted/40 hover:underline"
                title={row.path}
              >
                <BookOpen className="h-3.5 w-3.5 flex-none text-muted-foreground" />
                <span className="min-w-0 flex-1 truncate">{row.title}</span>
                <span className="flex items-center gap-1 font-mono text-[10px] text-muted-foreground">
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
