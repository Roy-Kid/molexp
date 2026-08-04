/**
 * Product action vocabulary — feature chrome should use these, not raw
 * `Button variant=…`. Visual mapping stays inside this module.
 */

import { Slot, Slottable } from "@radix-ui/react-slot";
import { RefreshCw, X } from "lucide-react";
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
  /** compact 28 · default 32 · comfortable 36 · content preserves caller layout */
  size?: "compact" | "default" | "comfortable" | "content";
  icon?: ReactNode;
  children?: ReactNode;
  asChild?: boolean;
}

const SIZE_MAP = {
  compact: "sm" as const,
  default: "default" as const,
  comfortable: "lg" as const,
  content: "content" as const,
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
    type={asChild ? undefined : type}
    variant={KIND_TO_VARIANT[kind]}
    size={SIZE_MAP[size]}
    className={cn(className)}
    asChild={asChild}
    {...rest}
  >
    {icon}
    {asChild ? <Slottable>{children}</Slottable> : children}
  </Button>
);

export interface WorkbenchIconActionProps
  extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "children"> {
  label: string;
  /** Icon controls are borderless: ghost by default, filled only for primary/destructive verbs. */
  kind?: Exclude<WorkbenchActionKind, "link" | "secondary">;
  size?: "compact" | "default";
  children: ReactNode;
  asChild?: boolean;
}

export const WorkbenchIconAction = ({
  label,
  kind = "ghost",
  size = "compact",
  children,
  className,
  asChild,
  type = "button",
  ...rest
}: WorkbenchIconActionProps): JSX.Element => (
  <Button
    type={asChild ? undefined : type}
    variant={KIND_TO_VARIANT[kind]}
    size="icon"
    aria-label={label}
    title={label}
    className={cn(size === "compact" ? "size-control-compact" : "size-control", className)}
    asChild={asChild}
    {...rest}
  >
    {children}
  </Button>
);

export type WorkbenchRetryActionProps = Omit<
  WorkbenchIconActionProps,
  "children" | "kind" | "label"
> & {
  label?: string;
};

/** Consistent borderless retry affordance for compact error and empty states. */
export const WorkbenchRetryAction = ({
  label = "Retry",
  ...rest
}: WorkbenchRetryActionProps): JSX.Element => (
  <WorkbenchIconAction label={label} kind="ghost" {...rest}>
    <RefreshCw aria-hidden />
  </WorkbenchIconAction>
);

export type WorkbenchDismissActionProps = Omit<
  WorkbenchIconActionProps,
  "children" | "kind" | "label"
> & {
  label?: string;
};

/** Consistent borderless dismiss affordance for inline notices. */
export const WorkbenchDismissAction = ({
  label = "Dismiss",
  ...rest
}: WorkbenchDismissActionProps): JSX.Element => (
  <WorkbenchIconAction label={label} kind="ghost" {...rest}>
    <X aria-hidden />
  </WorkbenchIconAction>
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
