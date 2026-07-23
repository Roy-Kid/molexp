/**
 * ReviewSurface — renders a harness FormDocument for step-audit approvals.
 *
 * Field shapes come from the OpenAPI pack (formDocument on PendingApprovalItem).
 * SchemaForm (run params) is intentionally not imported — different SoT.
 */

import { type JSX, useMemo, useState } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { MarkdownContent } from "@/components/ui/markdown";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { collectFieldValues, type FormDocumentWire, type FormFieldWire } from "./formDocument";

export type { FormDocumentWire, FormFieldWire } from "./formDocument";
export { collectFieldValues } from "./formDocument";

const FieldShell = ({
  field,
  children,
}: {
  field: FormFieldWire;
  children: React.ReactNode;
}): JSX.Element => (
  <div className="space-y-1">
    <Label className="text-xs font-medium text-foreground">
      {field.label}
      {field.required ? <span className="text-destructive"> *</span> : null}
    </Label>
    {field.help ? <p className="text-[11px] text-muted-foreground">{field.help}</p> : null}
    {children}
  </div>
);

export const ReviewSurface = ({
  formDocument,
  values,
  onChange,
  disabled,
}: {
  formDocument: FormDocumentWire | Record<string, unknown> | null | undefined;
  values: Record<string, unknown>;
  onChange: (next: Record<string, unknown>) => void;
  disabled?: boolean;
}): JSX.Element | null => {
  const doc = formDocument as FormDocumentWire | null | undefined;
  const fields = doc?.fields ?? [];
  if (!doc || fields.length === 0) return null;

  const set = (id: string, value: unknown): void => {
    onChange({ ...values, [id]: value });
  };

  return (
    <div className="space-y-3 rounded-md border border-border/60 bg-muted/20 px-3 py-2.5">
      {doc.title ? <h4 className="text-sm font-semibold text-foreground">{doc.title}</h4> : null}
      {doc.description_md ? (
        <div className="text-xs text-muted-foreground">
          <MarkdownContent text={doc.description_md} />
        </div>
      ) : null}
      {fields.map((field) => {
        const value = field.id in values ? values[field.id] : field.default;
        const ro = Boolean(field.readonly || disabled);

        if (field.kind === "markdown") {
          return (
            <FieldShell key={field.id} field={field}>
              <div className="text-xs">
                <MarkdownContent text={field.content ?? ""} />
              </div>
            </FieldShell>
          );
        }

        if (field.kind === "textarea") {
          return (
            <FieldShell key={field.id} field={field}>
              <Textarea
                value={String(value ?? "")}
                placeholder={field.placeholder}
                disabled={ro}
                rows={3}
                onChange={(e) => set(field.id, e.target.value)}
              />
            </FieldShell>
          );
        }

        if (field.kind === "number") {
          return (
            <FieldShell key={field.id} field={field}>
              <div className="flex items-center gap-2">
                <Input
                  type="number"
                  value={value === undefined || value === null ? "" : String(value)}
                  disabled={ro}
                  onChange={(e) =>
                    set(field.id, e.target.value === "" ? null : Number(e.target.value))
                  }
                />
                {field.unit ? (
                  <span className="text-xs text-muted-foreground">{field.unit}</span>
                ) : null}
              </div>
            </FieldShell>
          );
        }

        if (field.kind === "boolean") {
          return (
            <FieldShell key={field.id} field={field}>
              <label className="flex items-center gap-2 text-xs">
                <input
                  type="checkbox"
                  checked={Boolean(value)}
                  disabled={ro}
                  onChange={(e) => set(field.id, e.target.checked)}
                />
                <span>{value ? "Yes" : "No"}</span>
              </label>
            </FieldShell>
          );
        }

        if (field.kind === "select") {
          const options = field.options ?? [];
          return (
            <FieldShell key={field.id} field={field}>
              <Select
                value={String(value ?? "")}
                disabled={ro}
                onValueChange={(v) => set(field.id, v)}
              >
                <SelectTrigger className="h-8">
                  <SelectValue placeholder="Select…" />
                </SelectTrigger>
                <SelectContent>
                  {options.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value}>
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </FieldShell>
          );
        }

        if (field.kind === "multi_select") {
          const selected = Array.isArray(value) ? (value as string[]) : [];
          return (
            <FieldShell key={field.id} field={field}>
              <div className="flex flex-wrap gap-2">
                {(field.options ?? []).map((opt) => {
                  const on = selected.includes(opt.value);
                  return (
                    <label key={opt.value} className="flex items-center gap-1 text-xs">
                      <input
                        type="checkbox"
                        checked={on}
                        disabled={ro}
                        onChange={() => {
                          const next = on
                            ? selected.filter((v) => v !== opt.value)
                            : [...selected, opt.value];
                          set(field.id, next);
                        }}
                      />
                      {opt.label}
                    </label>
                  );
                })}
              </div>
            </FieldShell>
          );
        }

        if (field.kind === "table") {
          const rows = (field.default_rows ?? []) as Array<Record<string, unknown>>;
          const cols = field.columns ?? [];
          return (
            <FieldShell key={field.id} field={field}>
              <div className="overflow-x-auto">
                <table className="w-full border-collapse text-[11px]">
                  <thead>
                    <tr>
                      {cols.map((c) => (
                        <th
                          key={c.id}
                          className="border border-border/60 bg-muted/40 px-2 py-1 text-left"
                        >
                          {c.label}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {rows.length === 0 ? (
                      <tr>
                        <td
                          colSpan={Math.max(cols.length, 1)}
                          className="border border-border/60 px-2 py-1 text-muted-foreground"
                        >
                          (empty table — display only)
                        </td>
                      </tr>
                    ) : (
                      rows.map((row) => {
                        const rowKey = cols.map((c) => String(row[c.id] ?? "")).join("\0");
                        return (
                          <tr key={rowKey}>
                            {cols.map((c) => (
                              <td key={c.id} className="border border-border/60 px-2 py-1">
                                {String(row[c.id] ?? "")}
                              </td>
                            ))}
                          </tr>
                        );
                      })
                    )}
                  </tbody>
                </table>
              </div>
            </FieldShell>
          );
        }

        if (field.kind === "key_value") {
          return (
            <FieldShell key={field.id} field={field}>
              <pre className="max-h-32 overflow-auto rounded border border-border/60 bg-background px-2 py-1 font-mono text-[11px] text-muted-foreground">
                {JSON.stringify(value ?? field.default ?? [], null, 2)}
              </pre>
            </FieldShell>
          );
        }

        if (field.kind === "artifact_ref") {
          return (
            <FieldShell key={field.id} field={field}>
              <code className="text-[11px] text-muted-foreground">
                {String(value ?? field.default ?? "—")}
              </code>
            </FieldShell>
          );
        }

        // text + unknown kinds
        return (
          <FieldShell key={field.id} field={field}>
            <Input
              type="text"
              value={String(value ?? "")}
              placeholder={field.placeholder}
              disabled={ro}
              onChange={(e) => set(field.id, e.target.value)}
            />
          </FieldShell>
        );
      })}
    </div>
  );
};

/** Hook-friendly initial values from a form document. */
export const useFormDocumentValues = (
  formDocument: FormDocumentWire | Record<string, unknown> | null | undefined,
): [Record<string, unknown>, (next: Record<string, unknown>) => void] => {
  const initial = useMemo(
    () => collectFieldValues(formDocument as FormDocumentWire),
    [formDocument],
  );
  const [values, setValues] = useState<Record<string, unknown>>(initial);
  return [values, setValues];
};
