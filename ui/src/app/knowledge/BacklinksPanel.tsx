import { Link2 } from "lucide-react";
import type { JSX } from "react";
import type { NoteSummary } from "@/api/generated/models/NoteSummary";

interface BacklinksPanelProps {
  backlinks: NoteSummary[];
  /** Navigate to a linking document (its bundle-relative path). */
  onNavigate: (relPath: string) => void;
}

/**
 * Right-panel section listing every document that links at the selected Note.
 * Each entry is clickable and navigates to the linking document via the shared
 * `knowledge` selection mechanism.
 */
export const BacklinksPanel = ({ backlinks, onNavigate }: BacklinksPanelProps): JSX.Element => {
  return (
    <section className="space-y-2">
      <h3 className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        <Link2 className="h-3.5 w-3.5" /> Backlinks ({backlinks.length})
      </h3>
      {backlinks.length === 0 ? (
        <p className="text-xs italic text-muted-foreground">No documents link here yet.</p>
      ) : (
        <ul className="space-y-1">
          {backlinks.map((link) => (
            <li key={link.relPath}>
              <button
                type="button"
                onClick={() => onNavigate(link.relPath)}
                className="w-full truncate rounded-sm px-2 py-1 text-left text-sm text-info transition-colors hover:bg-muted/40 hover:underline"
                title={link.relPath}
              >
                {link.name}
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
};
