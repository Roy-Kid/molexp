import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import type * as React from "react";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
  // No border on any button — fill / ghost only; focus uses ring, not a stroke.
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-[var(--radius-control)] text-body font-medium transition-[color,background-color,box-shadow,opacity] duration-[var(--motion-base)] ease-[var(--motion-ease)] disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4 shrink-0 [&_svg]:shrink-0 outline-none border-0 focus-visible:ring-ring/50 focus-visible:ring-[3px] aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-[var(--molexp-accent-hover)]",
        destructive:
          "bg-destructive text-destructive-foreground hover:bg-[var(--status-failed-hover)] focus-visible:ring-destructive/20",
        outline:
          "bg-secondary text-secondary-foreground hover:bg-muted dark:bg-input/30 dark:hover:bg-input/50",
        secondary: "bg-secondary text-secondary-foreground hover:bg-muted",
        ghost: "hover:bg-interactive hover:text-interactive-foreground",
        link: "text-primary underline-offset-4 hover:underline",
      },
      size: {
        /* 32px default · 28px compact · 36px comfortable (constitution §2) */
        default: "h-control px-3 py-1 has-[>svg]:px-2",
        sm: "h-control-compact gap-1 px-2 text-label",
        lg: "h-control-comfortable rounded-[var(--radius-control)] px-4 has-[>svg]:px-3",
        icon: "size-control",
        "icon-sm": "size-control-compact",
        "icon-lg": "size-control-comfortable",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
);

function Button({
  className,
  variant,
  size,
  asChild = false,
  ...props
}: React.ComponentProps<"button"> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean;
  }) {
  const Comp = asChild ? Slot : "button";

  return (
    <Comp
      data-slot="button"
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  );
}

export { Button, buttonVariants };
