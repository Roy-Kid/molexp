/**
 * ParametersForm — a structured key/value editor for run parameters (the
 * workflow's root inputs), so users never hand-write JSON. Each row is a
 * `name = value` pair; values are coerced to the obvious JSON type (number /
 * boolean / null, else string) so e.g. `sigma = 1.0` becomes the number 1.0.
 *
 * The form owns its rows internally (initialised once from `value`); it calls
 * `onChange` with the rebuilt object on every edit. Mount it fresh (e.g. per
 * dialog open) when the initial value changes.
 */

import { Plus, X } from "lucide-react";
import { type JSX, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface Row {
  id: number;
  key: string;
  value: string;
}

/** Coerce a raw string cell to its obvious JSON type. */
function coerce(raw: string): unknown {
  const t = raw.trim();
  if (t === "") return "";
  if (t === "true") return true;
  if (t === "false") return false;
  if (t === "null") return null;
  // Numeric-looking → number (handles ints, floats, scientific: 10, 1.0, 1e-4).
  if (/^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(t)) {
    const n = Number(t);
    if (!Number.isNaN(n)) return n;
  }
  return raw;
}

function valueToCell(v: unknown): string {
  if (v === null || v === undefined) return "";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function toRows(value: Record<string, unknown>): Row[] {
  const rows = Object.entries(value).map(([key, v], i) => ({ id: i, key, value: valueToCell(v) }));
  return rows.length > 0 ? rows : [{ id: 0, key: "", value: "" }];
}

function toObject(rows: Row[]): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const row of rows) {
    const key = row.key.trim();
    if (key) out[key] = coerce(row.value);
  }
  return out;
}

export interface ParametersFormProps {
  value: Record<string, unknown>;
  onChange: (next: Record<string, unknown>) => void;
}

export function ParametersForm({ value, onChange }: ParametersFormProps): JSX.Element {
  const [rows, setRows] = useState<Row[]>(() => toRows(value));
  const [nextId, setNextId] = useState(() => toRows(value).length);

  const commit = (next: Row[]): void => {
    setRows(next);
    onChange(toObject(next));
  };

  const setRow = (id: number, patch: Partial<Row>): void =>
    commit(rows.map((r) => (r.id === id ? { ...r, ...patch } : r)));

  const addRow = (): void => {
    commit([...rows, { id: nextId, key: "", value: "" }]);
    setNextId(nextId + 1);
  };

  const removeRow = (id: number): void => {
    const next = rows.filter((r) => r.id !== id);
    commit(next.length > 0 ? next : [{ id: nextId, key: "", value: "" }]);
    if (next.length === 0) setNextId(nextId + 1);
  };

  return (
    <div className="space-y-1.5">
      {rows.map((row) => (
        <div key={row.id} className="flex items-center gap-1.5">
          <Input
            value={row.key}
            onChange={(e) => setRow(row.id, { key: e.target.value })}
            placeholder="name"
            className="h-8 flex-1 font-mono text-xs"
            aria-label="Parameter name"
          />
          <span className="text-muted-foreground">=</span>
          <Input
            value={row.value}
            onChange={(e) => setRow(row.id, { value: e.target.value })}
            placeholder="value"
            className="h-8 flex-1 font-mono text-xs"
            aria-label="Parameter value"
          />
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="h-7 w-7 flex-none text-muted-foreground hover:text-destructive"
            onClick={() => removeRow(row.id)}
            aria-label="Remove parameter"
          >
            <X className="h-3.5 w-3.5" />
          </Button>
        </div>
      ))}
      <Button
        type="button"
        variant="ghost"
        size="sm"
        className="h-7 gap-1 text-xs text-muted-foreground"
        onClick={addRow}
      >
        <Plus className="h-3.5 w-3.5" />
        Add parameter
      </Button>
    </div>
  );
}
