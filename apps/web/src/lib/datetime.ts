/**
 * The ONE human spelling for absolute timestamps across the app.
 *
 * Every surface that shows a wall-clock instant (experiment overview,
 * run details, execution records, asset metadata, …) renders it through
 * `formatDateTime` so the same moment never appears as
 * `02/07/2026, 18:16:01` in one card and `2026-07-02T18:16:01.039865`
 * in the next. Callers put the raw ISO string in the element's `title`
 * so hover reveals the precise value.
 */

const pad2 = (n: number): string => String(n).padStart(2, "0");

/**
 * Local-timezone "YYYY-MM-DD HH:mm" for an ISO timestamp.
 *
 * Returns "—" for null/undefined/empty and echoes the raw input when it
 * does not parse (never hides data behind a formatter failure).
 */
export const formatDateTime = (iso: string | null | undefined): string => {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const date = `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;
  const time = `${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
  return `${date} ${time}`;
};
