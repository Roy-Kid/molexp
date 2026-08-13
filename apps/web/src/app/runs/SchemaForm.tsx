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
import { ParameterField, type ParameterFieldDescriptor } from "@/components/workbench";

export type InputField = ParameterFieldDescriptor;

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

function initialValues(
  schema: InputField[],
  value: Record<string, unknown>,
): Record<string, unknown> {
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
    <div className="space-y-3">
      {schema.map((field) => (
        <ParameterField
          key={field.name}
          field={field}
          value={values[field.name]}
          onChange={(v) => set(field.name, v)}
        />
      ))}
      {schema.length === 0 && (
        <p className="text-label italic text-muted-foreground">This workflow declares no inputs.</p>
      )}
    </div>
  );
}
