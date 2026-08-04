import {
  Archive,
  ArrowUpRight,
  FileText,
  FlaskConical,
  NotebookPen,
  PlayCircle,
} from "lucide-react";
import type { ComponentType, JSX } from "react";
import type { EntityCard } from "@/api/generated/models/EntityCard";
import { StatusBadge } from "@/app/components/entity";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { Selection, WorkspaceSnapshot } from "@/app/types";
import { WorkbenchAction } from "@/components/workbench";
import { cn } from "@/lib/utils";

const KIND_ICON: Record<string, ComponentType<{ className?: string }>> = {
  run: PlayCircle,
  experiment: FlaskConical,
  asset: Archive,
  reference: FileText,
  note: NotebookPen,
};

const KIND_ICON_CLASS: Record<string, string> = {
  run: "text-muted-foreground",
  experiment: "text-muted-foreground",
  asset: "text-muted-foreground",
  reference: "text-muted-foreground",
  note: "text-muted-foreground",
};

/**
 * Map an embedded-entity card onto the navigation selection that opens it.
 * Runs / experiments / assets are entity ids; references and notes are OKF
 * Concepts reached by their bundle-relative path (the `knowledge` object type).
 * Returns `null` when the card carries nothing navigable so the card renders
 * disabled rather than dead-linking.
 */
const toSelection = (card: EntityCard): Selection | null => {
  switch (card.kind) {
    case "run":
      return { objectType: "run", objectId: card.id };
    case "experiment":
      return { objectType: "experiment", objectId: card.id };
    case "asset":
      return { objectType: "asset", objectId: card.id };
    case "reference":
    case "note":
      return card.relPath ? { objectType: "knowledge", objectId: card.relPath } : null;
    default:
      return card.relPath ? { objectType: "knowledge", objectId: card.relPath } : null;
  }
};

/**
 * A clickable inline card for an entity a document embeds. Renders the 06
 * summary card (kind / title / status) and, on click, reuses the LeftPanel
 * navigation via `useNavigationState().setSelection` to open the target entity —
 * no bespoke routing, so the URL + active selection update exactly as they do
 * from the sidebar.
 */
export const EntityRefCard = ({
  card,
  snapshot,
}: {
  card: EntityCard;
  snapshot: WorkspaceSnapshot;
}): JSX.Element => {
  const nav = useNavigationState(snapshot);
  const selection = toSelection(card);
  const Icon = KIND_ICON[card.kind] ?? FileText;

  return (
    <WorkbenchAction
      kind="ghost"
      size="content"
      type="button"
      disabled={selection === null}
      onClick={() => {
        if (selection) nav.setSelection(selection);
      }}
      className={cn(
        "group flex w-full items-center gap-3 rounded-control border border-border/60 bg-card px-3 py-2 text-left transition-colors",
        selection ? "hover:border-border hover:bg-muted/40" : "cursor-default opacity-70",
      )}
    >
      <Icon
        className={cn("h-4 w-4 flex-none", KIND_ICON_CLASS[card.kind] ?? "text-muted-foreground")}
      />
      <span className="min-w-0 flex-1">
        <span className="block truncate text-body-lg font-medium text-foreground">
          {card.title}
        </span>
        <span className="block truncate text-micro uppercase tracking-wide text-muted-foreground">
          {card.kind}
        </span>
      </span>
      {card.status && <StatusBadge status={card.status} size="sm" />}
      {selection && (
        <ArrowUpRight className="h-3.5 w-3.5 flex-none text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
      )}
    </WorkbenchAction>
  );
};
