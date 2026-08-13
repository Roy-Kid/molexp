/** Pure FormDocument helpers (no React / CSS) for step-audit ReviewSurface. */

export type FormFieldWire = {
  kind: string;
  id: string;
  label: string;
  help?: string;
  required?: boolean;
  readonly?: boolean;
  default?: unknown;
  placeholder?: string;
  content?: string;
  options?: Array<{ value: string; label: string }>;
  columns?: Array<{ id: string; label: string }>;
  default_rows?: Array<Record<string, unknown>>;
  unit?: string;
};

export type FormDocumentWire = {
  title?: string | null;
  description_md?: string | null;
  fields?: FormFieldWire[];
};

/** Collect editable field defaults / current values into a decision payload. */
export const collectFieldValues = (
  doc: FormDocumentWire | null | undefined,
  overrides: Record<string, unknown> = {},
): Record<string, unknown> => {
  const out: Record<string, unknown> = {};
  for (const field of doc?.fields ?? []) {
    if (field.readonly || field.kind === "markdown") continue;
    if (field.id in overrides) {
      out[field.id] = overrides[field.id];
    } else if (field.default !== undefined) {
      out[field.id] = field.default;
    }
  }
  return out;
};
