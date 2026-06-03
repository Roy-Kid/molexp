// ─────────────────────────────────────────────────────────────────────────────
// RelatedPanel — renders the relation groups for the current entity as grouped,
// clickable rows. Dropped into the right inspector, it gives every entity a
// consistent "what is this connected to, and jump there" surface, replacing the
// ad-hoc per-viewer ``setSelection`` links that were inconsistent or missing.
// ─────────────────────────────────────────────────────────────────────────────

import type { JSX } from "react";
import { useNavigate } from "react-router-dom";
import { StatusBadge } from "@/app/components/entity";
import { type EntityRef, entityMeta } from "@/app/entities/kinds";
import { entityPath } from "@/app/entities/paths";
import { resolveRelations } from "@/app/entities/relations";
import type { SemanticStatus, WorkspaceSnapshot } from "@/app/types";

interface RelatedPanelProps {
  entity: EntityRef;
  snapshot: WorkspaceSnapshot;
}

const RelatedRow = ({
  refItem,
  snapshot,
}: {
  refItem: EntityRef;
  snapshot: WorkspaceSnapshot;
}): JSX.Element => {
  const navigate = useNavigate();
  const meta = entityMeta(refItem.kind);
  const Icon = meta.icon;
  const path = entityPath(refItem, snapshot);

  return (
    <button
      type="button"
      disabled={!path}
      onClick={() => {
        if (path) navigate(path);
      }}
      className="group flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-left transition-colors hover:bg-muted/60 disabled:opacity-40"
    >
      <Icon className={`h-3.5 w-3.5 flex-none ${meta.accent}`} />
      <span className="min-w-0 flex-1 truncate text-xs text-foreground">
        {refItem.label ?? refItem.id}
      </span>
      {refItem.status && (
        <StatusBadge status={refItem.status as SemanticStatus} size="sm" dot showLabel={false} />
      )}
    </button>
  );
};

export const RelatedPanel = ({ entity, snapshot }: RelatedPanelProps): JSX.Element | null => {
  const groups = resolveRelations(entity, snapshot);
  if (groups.length === 0) {
    return null;
  }

  return (
    <section className="border-t border-border/70 px-2 py-3">
      <h3 className="px-2 pb-1.5 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
        Related
      </h3>
      <div className="space-y-2.5">
        {groups.map((g) => (
          <div key={g.relation}>
            <p className="px-2 pb-0.5 text-[10px] font-medium uppercase tracking-wide text-muted-foreground/70">
              {g.label}
              {g.refs.length > 1 && (
                <span className="ml-1 tabular-nums opacity-60">{g.refs.length}</span>
              )}
            </p>
            {g.refs.map((refItem) => (
              <RelatedRow
                key={`${refItem.kind}:${refItem.id}`}
                refItem={refItem}
                snapshot={snapshot}
              />
            ))}
          </div>
        ))}
      </div>
    </section>
  );
};
