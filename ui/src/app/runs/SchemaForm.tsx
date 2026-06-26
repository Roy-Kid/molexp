/**
 * SchemaForm — a *typed* run-input form driven by a workflow's declared input
 * schema (derived server-side from the tasks' typed parameters). Each field
 * renders the right widget for its type: number input for `number`/`integer`,
 * a checkbox for `boolean`, an enum dropdown for `enum`, a text field otherwise.
 *
 * This is the "normal form" — fixed, labelled fields — as opposed to the
 * free-form key/value {@link ParametersForm} used when a workflow declares no
 * schema.
 */

import { type JSX, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export interface InputField {
  name: string;
  type: "number" | "integer" | "text" | "boolean" | "enum";
  default: unknown;
  options?: (string | number)[];
}

/**
 * Read the workflow's declared input schema out of an experiment's workflow IR
 * string (the server derives it from the tasks' typed parameters). Returns
 * `null` when the IR is absent/unparseable or declares no schema — callers then
 * fall back to the free-form key/value form.
 */
export function parseInputSchema(workflowIr: string | null | undefined): InputField[] | null {
  if (!workflowIr) return null;
  try {
    const parsed = JSON.parse(workflowIr) as { input_schema?: unknown };
    const schema = parsed.input_schema;
    if (Array.isArray(schema) && schema.length > 0) return schema as InputField[];
  } catch {
    return null;
  }
  return null;
}

/** Default values for a schema, as the initial run parameters. */
export function schemaDefaults(schema: InputField[]): Record<string, unknown> {
  return Object.fromEntries(schema.map((f) => [f.name, f.default]));
}

export interface SchemaFormProps {
  schema: InputField[];
  value: Record<string, unknown>;
  onChange: (next: Record<string, unknown>) => void;
}

function initialValues(schema: InputField[], value: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const field of schema) {
    out[field.name] = field.name in value ? value[field.name] : field.default;
  }
  return out;
}

export function SchemaForm({ schema, value, onChange }: SchemaFormProps): JSX.Element {
  const [values, setValues] = useState<Record<string, unknown>>(() => initialValues(schema, value));

  const set = (name: string, v: unknown): void => {
    const next = { ...values, [name]: v };
    setValues(next);
    onChange(next);
  };

  return (
    <div className="space-y-2.5">
      {schema.map((field) => {
        const current = values[field.name];
        return (
          <div
            key={field.name}
            className="grid grid-cols-1 items-center gap-1 sm:grid-cols-3 sm:gap-3"
          >
            <Label htmlFor={`f-${field.name}`} className="font-mono text-xs sm:text-right">
              {field.name}
            </Label>
            <div className="sm:col-span-2">
              {field.type === "enum" ? (
                <Select
                  value={String(current ?? "")}
                  onValueChange={(v) => set(field.name, v)}
                >
                  <SelectTrigger id={`f-${field.name}`} className="h-8">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {(field.options ?? []).map((opt) => (
                      <SelectItem key={String(opt)} value={String(opt)}>
                        {String(opt)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              ) : field.type === "boolean" ? (
                <input
                  id={`f-${field.name}`}
                  type="checkbox"
                  checked={Boolean(current)}
                  onChange={(e) => set(field.name, e.target.checked)}
                  className="h-4 w-4 rounded border-border"
                />
              ) : field.type === "number" || field.type === "integer" ? (
                <Input
                  id={`f-${field.name}`}
                  type="number"
                  step={field.type === "integer" ? "1" : "any"}
                  value={current === null || current === undefined ? "" : String(current)}
                  onChange={(e) => {
                    const raw = e.target.value;
                    if (raw === "") return set(field.name, null);
                    const n = field.type === "integer" ? parseInt(raw, 10) : Number(raw);
                    set(field.name, Number.isNaN(n) ? raw : n);
                  }}
                  className="h-8 font-mono text-xs"
                />
              ) : (
                <Input
                  id={`f-${field.name}`}
                  type="text"
                  value={current === null || current === undefined ? "" : String(current)}
                  onChange={(e) => set(field.name, e.target.value)}
                  className="h-8 font-mono text-xs"
                />
              )}
            </div>
          </div>
        );
      })}
      {schema.length === 0 && (
        <p className="text-xs italic text-muted-foreground">This workflow declares no inputs.</p>
      )}
      <span className="sr-only">
        <Button type="button" tabIndex={-1} />
      </span>
    </div>
  );
}
