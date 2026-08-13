/**
 * Schema-driven parameter control — one field from a workflow input schema.
 * Visual mapping lives here; call sites pass domain field descriptors only.
 */

import { type JSX, useId } from "react";

import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

export interface ParameterFieldDescriptor {
  name: string;
  label?: string;
  description?: string;
  unit?: string;
  required?: boolean;
  type: "number" | "integer" | "text" | "boolean" | "enum";
  default: unknown;
  options?: (string | number)[];
}

export interface ParameterFieldProps {
  field: ParameterFieldDescriptor;
  value: unknown;
  onChange: (value: unknown) => void;
  error?: string | null;
  disabled?: boolean;
  className?: string;
}

export const ParameterField = ({
  field,
  value,
  onChange,
  error,
  disabled,
  className,
}: ParameterFieldProps): JSX.Element => {
  const generatedId = useId();
  const id = `parameter-${generatedId}`;
  const detailId = `${id}-detail`;
  const hasDetail = Boolean(error || field.description);
  const label = field.label ?? field.name;

  return (
    <div
      className={cn("grid grid-cols-(--form-grid-columns) items-start gap-x-3 gap-y-1", className)}
    >
      <Label htmlFor={id} className="pt-2 text-label font-normal text-muted-foreground">
        {label}
        {field.required && <span aria-hidden="true">·</span>}
      </Label>
      <div className="min-w-0 space-y-1">
        {field.type === "enum" ? (
          <Select
            value={String(value ?? "")}
            onValueChange={(next) => {
              const option = field.options?.find((candidate) => String(candidate) === next);
              onChange(option ?? next);
            }}
            disabled={disabled}
          >
            <SelectTrigger
              id={id}
              className="w-full text-body"
              aria-invalid={Boolean(error)}
              aria-describedby={hasDetail ? detailId : undefined}
            >
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
          <div className="flex h-control items-center gap-2 text-body">
            <Checkbox
              id={id}
              checked={Boolean(value)}
              disabled={disabled}
              required={field.required}
              aria-invalid={Boolean(error)}
              aria-describedby={hasDetail ? detailId : undefined}
              onCheckedChange={(checked) => onChange(Boolean(checked))}
            />
            <span className="font-mono text-label text-muted-foreground">
              {value ? "true" : "false"}
            </span>
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <Input
              id={id}
              type={field.type === "number" || field.type === "integer" ? "number" : "text"}
              step={field.type === "integer" ? 1 : "any"}
              className={cn("font-mono tabular-nums", field.type === "text" && "font-sans")}
              value={value === null || value === undefined ? "" : String(value)}
              disabled={disabled}
              required={field.required}
              aria-invalid={Boolean(error)}
              aria-describedby={hasDetail ? detailId : undefined}
              onChange={(event) => {
                const raw = event.target.value;
                if (field.type === "number" || field.type === "integer") {
                  if (raw === "" || raw === "-") {
                    onChange(raw);
                    return;
                  }
                  const n = field.type === "integer" ? Number.parseInt(raw, 10) : Number(raw);
                  onChange(Number.isNaN(n) ? raw : n);
                  return;
                }
                onChange(raw);
              }}
            />
            {field.unit && (
              <span className="flex-none font-mono text-label text-muted-foreground">
                {field.unit}
              </span>
            )}
          </div>
        )}
        {hasDetail && (
          <p
            id={detailId}
            role={error ? "alert" : undefined}
            className={cn(
              "text-micro",
              error ? "text-status-failed-foreground" : "text-muted-foreground",
            )}
          >
            {error ?? field.description}
          </p>
        )}
      </div>
    </div>
  );
};
