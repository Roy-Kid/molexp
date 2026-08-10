// ─────────────────────────────────────────────────────────────────────────────
// RelatedPanel — right-inspector graph edges for the current entity.
// Lineage (ancestors / provenance) sits above Related (children, siblings).
// Center overviews never re-host these jumps — navigation lives here.
// ─────────────────────────────────────────────────────────────────────────────

import type { JSX, ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { StatusBadge } from "@/app/components/entity";
import { type EntityRef, entityMeta } from "@/app/entities/kinds";
import { entityPath } from "@/app/entities/paths";
import { LINEAGE_RELATIONS, resolveRelations } from "@/app/entities/relations";
import type { SemanticStatus, WorkspaceSnapshot } from "@/app/types";
import { WorkbenchAction } from "@/components/workbench";

interface RelatedPanelProps {
  entity: EntityRef;
  snapshot: WorkspaceSnapshot;
}

const RelatedRow = ({
  refItem,
  snapshot,
  roleLabel,
}: {
  refItem: EntityRef;
  snapshot: WorkspaceSnapshot;
  /** Optional relation label when the row is a lineage chain step. */
  roleLabel?: string;
}): JSX.Element => {
  const navigate = useNavigate();
  const meta = entityMeta(refItem.kind);
  const Icon = meta.icon;
  const path = entityPath(refItem, snapshot);

  return (
    <WorkbenchAction
      kind="ghost"
      size="content"
      type="button"
      disabled={!path}
      onClick={() => {
        if (path) navigate(path);
      }}
      className="group flex w-full items-center gap-2 rounded-control px-2 py-1.5 text-left transition-colors hover:bg-interactive/60 disabled:opacity-40"
    >
      <Icon className={`size-3.5 flex-none ${meta.iconClassName}`} aria-hidden />
      <span className="flex min-w-0 flex-1 flex-col">
        {roleLabel && <span className="text-micro text-muted-foreground">{roleLabel}</span>}
        <span className="min-w-0 truncate text-label text-foreground">
          {refItem.label ?? refItem.id}
        </span>
      </span>
      {refItem.status && (
        <StatusBadge status={refItem.status as SemanticStatus} size="sm" dot showLabel={false} />
      )}
    </WorkbenchAction>
  );
};

const Section = ({ title, children }: { title: string; children: ReactNode }): JSX.Element => (
  <section className="border-t border-border px-2 py-2">
    <h3 className="px-2 pb-1.5 text-micro font-medium uppercase tracking-wide text-muted-foreground">
      {title}
    </h3>
    {children}
  </section>
);

export const RelatedPanel = ({ entity, snapshot }: RelatedPanelProps): JSX.Element | null => {
  const groups = resolveRelations(entity, snapshot);
  if (groups.length === 0) {
    return null;
  }

  const lineageOrder = new Map(LINEAGE_RELATIONS.map((key, index) => [key, index]));
  const lineageGroups = groups
    .filter((g) => lineageOrder.has(g.relation))
    .sort((a, b) => (lineageOrder.get(a.relation) ?? 0) - (lineageOrder.get(b.relation) ?? 0));
  const relatedGroups = groups.filter((g) => !lineageOrder.has(g.relation));

  if (lineageGroups.length === 0 && relatedGroups.length === 0) {
    return null;
  }

  return (
    <div className="flex flex-col">
      {lineageGroups.length > 0 && (
        <Section title="Lineage">
          <div className="space-y-0.5">
            {lineageGroups.flatMap((g) =>
              g.refs.map((refItem) => (
                <RelatedRow
                  key={`${g.relation}:${refItem.kind}:${refItem.id}`}
                  refItem={refItem}
                  snapshot={snapshot}
                  roleLabel={g.label}
                />
              )),
            )}
          </div>
        </Section>
      )}

      {relatedGroups.map((g) => (
        <Section key={g.relation} title={g.label}>
          <div className="space-y-0.5">
            {g.refs.map((refItem) => (
              <RelatedRow
                key={`${refItem.kind}:${refItem.id}`}
                refItem={refItem}
                snapshot={snapshot}
              />
            ))}
            {g.refs.length === 0 && (
              <p className="px-2 py-1 text-micro text-muted-foreground">None</p>
            )}
          </div>
        </Section>
      ))}
    </div>
  );
};
