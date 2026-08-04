/**
 * Product-owned scientific data colors.
 *
 * These are intentionally separate from brand and run-status colors. Charts
 * need concrete color strings because canvas renderers cannot reliably resolve
 * CSS custom properties; centralizing them here keeps feature components free
 * of palette literals.
 */
export const CHART_SERIES_PALETTE = [
  "oklch(0.55 0.18 255)",
  "oklch(0.55 0.19 25)",
  "oklch(0.58 0.14 150)",
  "oklch(0.68 0.14 70)",
  "oklch(0.55 0.19 295)",
  "oklch(0.6 0.12 210)",
  "oklch(0.58 0.18 350)",
  "oklch(0.62 0.14 130)",
] as const;

export const CHART_CLUSTER_PALETTE = [
  "oklch(0.58 0.16 250)",
  "oklch(0.55 0.19 295)",
  "oklch(0.6 0.13 150)",
  "oklch(0.72 0.14 85)",
  "oklch(0.58 0.19 25)",
  "oklch(0.62 0.12 210)",
  "oklch(0.58 0.16 320)",
  "oklch(0.65 0.14 130)",
] as const;

export const DELTA_F_GROUP_COLORS = {
  inference: "oklch(0.52 0.14 250)",
  training: "oklch(0.72 0.16 70)",
  combined: "oklch(0.55 0.14 310)",
} as const;
