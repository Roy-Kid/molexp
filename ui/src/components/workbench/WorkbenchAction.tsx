/**
 * Product action vocabulary — feature chrome should use these, not raw
 * `Button variant=…`. Visual mapping stays inside this module.
 */

import { Slot } from "@radix-ui/react-slot";
import type { ButtonHTMLAttributes, JSX, ReactNode } from "react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export type WorkbenchActionKind = "primary" | "secondary" | "ghost" | "danger" | "link";

const KIND_TO_VARIANT: Record<
  WorkbenchActionKind,
  "default" | "outline" | "ghost" | "destructive" | "link"
> = {
  primary: "default",
  secondary: "outline",
  ghost: "ghost",
  danger: "destructive",
  link: "link",
};

export interface WorkbenchActionProps
  extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "children"> {
  kind?: WorkbenchActionKind;
  /** compact 28 · default 32 · comfortable 36 */
  size?: "compact" | "default" | "comfortable";
  icon?: ReactNode;
  children?: ReactNode;
  asChild?: boolean;
}

const SIZE_MAP = {
  compact: "sm" as const,
  default: "default" as const,
  comfortable: "lg" as const,
};

export const WorkbenchAction = ({
  kind = "secondary",
  size = "default",
  icon,
  children,
  className,
  asChild,
  type = "button",
  ...rest
}: WorkbenchActionProps): JSX.Element => (
  <Button
    type={type}
    variant={KIND_TO_VARIANT[kind]}
    size={SIZE_MAP[size]}
    className={cn(className)}
    asChild={asChild}
    {...rest}
  >
    {icon}
    {children}
  </Button>
);

export interface WorkbenchIconActionProps
  extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "children"> {
  label: string;
  kind?: WorkbenchActionKind;
  size?: "compact" | "default";
  children: ReactNode;
}

export const WorkbenchIconAction = ({
  label,
  kind = "ghost",
  size = "compact",
  children,
  className,
  type = "button",
  ...rest
}: WorkbenchIconActionProps): JSX.Element => (
  <Button
    type={type}
    variant={KIND_TO_VARIANT[kind]}
    size="icon"
    aria-label={label}
    title={label}
    className={cn(size === "compact" ? "size-control-compact" : "size-control", className)}
    {...rest}
  >
    {children}
  </Button>
);

/** Tooltip-friendly icon control without importing Button variants in features. */
export const WorkbenchToggleAction = ({
  pressed,
  label,
  children,
  className,
  ...rest
}: {
  pressed: boolean;
  label: string;
  children: ReactNode;
} & Omit<ButtonHTMLAttributes<HTMLButtonElement>, "children" | "aria-pressed">): JSX.Element => (
  <Button
    type="button"
    variant={pressed ? "secondary" : "ghost"}
    size="icon"
    aria-label={label}
    aria-pressed={pressed}
    title={label}
    className={cn("size-control-compact", className)}
    {...rest}
  >
    {children}
  </Button>
);

// re-export Slot for rare asChild composition without pulling radix in features
export { Slot };
