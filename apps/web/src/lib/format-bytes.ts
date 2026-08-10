/**
 * Shared byte-size formatter for asset manifests, dataset file trees, and
 * any other size column. Binary units (1 KB = 1024 B), one decimal place
 * past the byte threshold, capped at GB. Returns "—" for nullish / non-finite
 * input so callers can render it directly.
 */
export const formatBytes = (bytes: number | null | undefined): string => {
  if (bytes == null || !Number.isFinite(bytes)) return "—";
  const value = Number(bytes);
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  if (value < 1024 * 1024 * 1024) return `${(value / (1024 * 1024)).toFixed(1)} MB`;
  return `${(value / (1024 * 1024 * 1024)).toFixed(1)} GB`;
};
