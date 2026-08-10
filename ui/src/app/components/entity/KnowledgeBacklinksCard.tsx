import { BookOpen, Link2 } from "lucide-react";
import { type JSX, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import type { EntityBacklinkRow } from "@/api/generated/models/EntityBacklinkRow";
import { KnowledgeService } from "@/api/generated/services/KnowledgeService";
import { DashboardCard } from "@/app/components/entity/Dashboard";
import { WorkbenchAction } from "@/components/workbench";

interface KnowledgeBacklinksCardProps {
  kind: "run" | "experiment";
  projectId: string;
  experimentId: string;
  runId?: string;
  className?: string;
}

/**
 * Knowledge documents citing this entity. Renders only when backlinks exist —
 * never reserves overview fold for empty/loading chrome.
 */
export const KnowledgeBacklinksCard = ({
  kind,
  projectId,
  experimentId,
  runId,
  className,
}: KnowledgeBacklinksCardProps): JSX.Element | null => {
  const navigate = useNavigate();
  const requestKey = `${kind}:${projectId}:${experimentId}:${runId ?? ""}`;
  const [rows, setRows] = useState<EntityBacklinkRow[] | null>(null);
  const [settledRequestKey, setSettledRequestKey] = useState("");

  useEffect(() => {
    let cancelled = false;
    setRows(null);
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
      })
      .finally(() => {
        if (!cancelled) setSettledRequestKey(requestKey);
      });
    return () => {
      cancelled = true;
    };
  }, [kind, projectId, experimentId, runId, requestKey]);

  if (settledRequestKey !== requestKey || !rows || rows.length === 0) {
    return null;
  }

  return (
    <DashboardCard
      title="Knowledge"
      icon={BookOpen}
      description={kind === "run" ? "Notes citing this run" : "Notes citing this experiment"}
      className={className}
    >
      <ul className="space-y-1">
        {rows.map((row) => (
          <li key={row.path}>
            <WorkbenchAction
              kind="ghost"
              size="content"
              type="button"
              onClick={() =>
                navigate(`/knowledge/${row.path.split("/").map(encodeURIComponent).join("/")}`)
              }
              className="flex w-full items-center gap-2 truncate rounded-control px-2 py-2 text-left text-body-lg transition-colors hover:bg-muted/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              title={row.path}
            >
              <BookOpen className="h-3.5 w-3.5 flex-none text-muted-foreground" />
              <span className="min-w-0 flex-1 truncate text-foreground">{row.title}</span>
              <span className="inline-flex items-center gap-1 font-mono text-micro text-muted-foreground">
                <Link2 className="h-3 w-3" />
                {row.role}
              </span>
            </WorkbenchAction>
          </li>
        ))}
      </ul>
    </DashboardCard>
  );
};
