/**
 * ReviewSurface — renders a harness FormDocument for step-audit approvals.
 *
 * Field shapes come from the OpenAPI pack (formDocument on PendingApprovalItem).
 * SchemaForm (run params) is intentionally not imported — different SoT.
 */

import { type JSX, useId, useMemo, useState } from "react";
import { Checkbox } from "@/components/ui/checkbox";
import { Code as InlineCode } from "@/components/ui/code";
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
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import { collectFieldValues, type FormDocumentWire, type FormFieldWire } from "./formDocument";

export type { FormDocumentWire, FormFieldWire } from "./formDocument";
export { collectFieldValues } from "./formDocument";

const FieldShell = ({
  field,
  controlId,
  helpId,
  children,
  hideHelp,
}: {
  field: FormFieldWire;
  controlId?: string;
  helpId: string;
  children: React.ReactNode;
  hideHelp?: boolean;
}): JSX.Element => (
  <div className="space-y-1">
    {controlId ? (
      <Label htmlFor={controlId} className="text-label font-medium text-foreground">
        {field.label}
        {field.required ? <span className="text-destructive"> *</span> : null}
      </Label>
    ) : (
      <p className="text-label font-medium text-foreground">
        {field.label}
        {field.required ? <span className="text-destructive"> *</span> : null}
      </p>
    )}
    {!hideHelp && field.help ? (
      <p id={helpId} className="text-micro text-muted-foreground">
        {field.help}
      </p>
    ) : null}
    {children}
  </div>
);

export const ReviewSurface = ({
  formDocument,
  values,
  onChange,
  disabled,
  compact = false,
}: {
  formDocument: FormDocumentWire | Record<string, unknown> | null | undefined;
  values: Record<string, unknown>;
  onChange: (next: Record<string, unknown>) => void;
  disabled?: boolean;
  /** Parent already shows title/blurb — skip repeating them here. */
  compact?: boolean;
}): JSX.Element | null => {
  const surfaceId = useId();
  const doc = formDocument as FormDocumentWire | null | undefined;
  const fields = doc?.fields ?? [];
  if (!doc || fields.length === 0) return null;

  const set = (id: string, value: unknown): void => {
    onChange({ ...values, [id]: value });
  };

  return (
    <div className={cn("space-y-3", !compact && "border-t border-border/60 pt-3")}>
      {!compact && doc.title ? (
        <h4 className="text-body-lg font-semibold text-foreground">{doc.title}</h4>
      ) : null}
      {!compact && doc.description_md ? (
        <div className="text-label text-muted-foreground">
          <MarkdownContent text={doc.description_md} />
        </div>
      ) : null}
      {fields.map((field) => {
        const value = field.id in values ? values[field.id] : field.default;
        const ro = Boolean(field.readonly || disabled);
        const safeFieldId = field.id.replace(/[^a-zA-Z0-9_-]/g, "-");
        const controlId = `${surfaceId}-${safeFieldId}`;
        const helpId = `${controlId}-help`;
        const describedBy = field.help && !compact ? helpId : undefined;

        if (field.kind === "markdown") {
          return (
            <FieldShell key={field.id} field={field} helpId={helpId} hideHelp={compact}>
              <div className="text-label leading-relaxed text-foreground">
                <MarkdownContent text={field.content ?? ""} />
              </div>
            </FieldShell>
          );
        }

        if (field.kind === "textarea") {
          return (
            <FieldShell key={field.id} field={field} controlId={controlId} helpId={helpId}>
              <Textarea
                id={controlId}
                value={String(value ?? "")}
                placeholder={field.placeholder}
                disabled={ro}
                required={field.required}
                aria-describedby={describedBy}
                rows={3}
                onChange={(e) => set(field.id, e.target.value)}
              />
            </FieldShell>
          );
        }

        if (field.kind === "number") {
          return (
            <FieldShell key={field.id} field={field} controlId={controlId} helpId={helpId}>
              <div className="flex items-center gap-2">
                <Input
                  id={controlId}
                  type="number"
                  value={value === undefined || value === null ? "" : String(value)}
                  disabled={ro}
                  required={field.required}
                  aria-describedby={describedBy}
                  onChange={(e) =>
                    set(field.id, e.target.value === "" ? null : Number(e.target.value))
                  }
                />
                {field.unit ? (
                  <span className="text-label text-muted-foreground">{field.unit}</span>
                ) : null}
              </div>
            </FieldShell>
          );
        }

        if (field.kind === "boolean") {
          return (
            <FieldShell key={field.id} field={field} controlId={controlId} helpId={helpId}>
              <div className="flex items-center gap-2 text-label">
                <Checkbox
                  id={controlId}
                  checked={Boolean(value)}
                  disabled={ro}
                  required={field.required}
                  aria-describedby={describedBy}
                  onCheckedChange={(checked) => set(field.id, Boolean(checked))}
                />
                <span>{value ? "Yes" : "No"}</span>
              </div>
            </FieldShell>
          );
        }

        if (field.kind === "select") {
          const options = field.options ?? [];
          // Radix Select forbids empty-string values — use undefined for uncontrolled placeholder.
          const selectValue =
            value === undefined || value === null || value === "" ? undefined : String(value);
          return (
            <FieldShell key={field.id} field={field} controlId={controlId} helpId={helpId}>
              <Select value={selectValue} disabled={ro} onValueChange={(v) => set(field.id, v)}>
                <SelectTrigger
                  id={controlId}
                  className="w-full min-w-panel-sm"
                  aria-describedby={describedBy}
                >
                  <SelectValue placeholder="Select…" />
                </SelectTrigger>
                <SelectContent>
                  {options.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value || "__empty__"}>
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
            <FieldShell key={field.id} field={field} helpId={helpId} hideHelp={compact}>
              <ul className="space-y-1.5">
                {(field.options ?? []).map((opt, index) => {
                  const on = selected.includes(opt.value);
                  const optionId = `${controlId}-${index}`;
                  return (
                    <li key={opt.value}>
                      <label
                        htmlFor={optionId}
                        className="flex cursor-pointer items-start gap-2 rounded-control px-2 py-1.5 text-body-lg hover:bg-muted/50"
                      >
                        <Checkbox
                          id={optionId}
                          className="mt-0.5"
                          checked={on}
                          disabled={ro}
                          aria-describedby={describedBy}
                          onCheckedChange={() => {
                            const next = on
                              ? selected.filter((v) => v !== opt.value)
                              : [...selected, opt.value];
                            set(field.id, next);
                          }}
                        />
                        <span className="leading-snug text-foreground">{opt.label}</span>
                      </label>
                    </li>
                  );
                })}
              </ul>
            </FieldShell>
          );
        }

        if (field.kind === "table") {
          const rows = (field.default_rows ?? []) as Array<Record<string, unknown>>;
          const cols = field.columns ?? [];
          return (
            <FieldShell key={field.id} field={field} helpId={helpId}>
              <div className="overflow-x-auto">
                <Table className="w-full border-collapse text-micro">
                  <TableHeader>
                    <TableRow>
                      {cols.map((c) => (
                        <TableHead
                          key={c.id}
                          className="border border-border/60 bg-muted/40 px-2 py-1 text-left"
                        >
                          {c.label}
                        </TableHead>
                      ))}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {rows.length === 0 ? (
                      <TableRow>
                        <TableCell
                          colSpan={Math.max(cols.length, 1)}
                          className="border border-border/60 px-2 py-1 text-muted-foreground"
                        >
                          (empty table — display only)
                        </TableCell>
                      </TableRow>
                    ) : (
                      rows.map((row) => {
                        const rowKey = cols.map((c) => String(row[c.id] ?? "")).join("\0");
                        return (
                          <TableRow key={rowKey}>
                            {cols.map((c) => (
                              <TableCell key={c.id} className="border border-border/60 px-2 py-1">
                                {String(row[c.id] ?? "")}
                              </TableCell>
                            ))}
                          </TableRow>
                        );
                      })
                    )}
                  </TableBody>
                </Table>
              </div>
            </FieldShell>
          );
        }

        if (field.kind === "key_value") {
          return (
            <FieldShell key={field.id} field={field} helpId={helpId}>
              <pre className="max-h-32 overflow-auto rounded-control border border-border/60 bg-background px-2 py-1 font-mono text-micro text-muted-foreground">
                {JSON.stringify(value ?? field.default ?? [], null, 2)}
              </pre>
            </FieldShell>
          );
        }

        if (field.kind === "artifact_ref") {
          return (
            <FieldShell key={field.id} field={field} helpId={helpId}>
              <InlineCode className="text-micro text-muted-foreground">
                {String(value ?? field.default ?? "—")}
              </InlineCode>
            </FieldShell>
          );
        }

        // text + unknown kinds
        return (
          <FieldShell key={field.id} field={field} controlId={controlId} helpId={helpId}>
            <Input
              id={controlId}
              type="text"
              value={String(value ?? "")}
              placeholder={field.placeholder}
              disabled={ro}
              required={field.required}
              aria-describedby={describedBy}
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
