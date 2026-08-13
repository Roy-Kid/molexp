/**
 * Human-facing labels for molplot host files.
 *
 * On disk names are gated by suffixes (``*.mlp.vl.json``, ``*.mlp.jsonl``, …);
 * users should never see that machinery in the UI.
 */

/** Longest-first so ``.mlp.vl.json`` wins over a bare ``.json``. */
const MOLPLOT_SUFFIXES = [
  ".mlp.vl.json",
  ".mlp.index.json",
  ".mlp.jsonl",
  ".mlp.zarr",
  ".mlp.json",
] as const;

/**
 * Strip molplot host suffixes from a file *basename* for display.
 *
 * ``nve_energy.mlp.vl.json`` → ``nve_energy``
 * ``metrics.mlp.jsonl`` → ``metrics``
 */
export function molplotDisplayName(name: string): string {
  const base = name.trim();
  if (!base) return base;
  // Take last path segment if a rel path slipped in.
  const leaf = base.includes("/") ? base.slice(base.lastIndexOf("/") + 1) : base;
  const lower = leaf.toLowerCase();
  for (const suffix of MOLPLOT_SUFFIXES) {
    if (lower.endsWith(suffix)) {
      const stripped = leaf.slice(0, -suffix.length);
      return stripped || leaf;
    }
  }
  // Dense zarr dir sometimes surfaces as ``zarr.json`` under ``*.mlp.zarr/``.
  if (lower === "zarr.json") return "metrics";
  return leaf;
}
