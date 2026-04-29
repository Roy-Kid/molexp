import type { ReactNode } from "react";
import { useCallback, useState } from "react";

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { cn } from "@/lib/utils";

export interface ConfirmOptions {
  title: ReactNode;
  description?: ReactNode;
  confirmLabel?: string;
  cancelLabel?: string;
  destructive?: boolean;
}

interface InternalState extends ConfirmOptions {
  resolve: (value: boolean) => void;
}

interface UseConfirmReturn {
  confirm: (options: ConfirmOptions) => Promise<boolean>;
  dialog: ReactNode;
}

export const useConfirm = (): UseConfirmReturn => {
  const [state, setState] = useState<InternalState | null>(null);

  const confirm = useCallback(
    (options: ConfirmOptions): Promise<boolean> =>
      new Promise<boolean>((resolve) => {
        setState({ ...options, resolve });
      }),
    [],
  );

  const close = (value: boolean): void => {
    if (state) {
      state.resolve(value);
      setState(null);
    }
  };

  const dialog = (
    <AlertDialog
      open={state !== null}
      onOpenChange={(open) => {
        if (!open) close(false);
      }}
    >
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{state?.title}</AlertDialogTitle>
          {state?.description !== undefined && (
            <AlertDialogDescription asChild>
              <div>{state.description}</div>
            </AlertDialogDescription>
          )}
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel onClick={() => close(false)}>
            {state?.cancelLabel ?? "Cancel"}
          </AlertDialogCancel>
          <AlertDialogAction
            className={cn(
              state?.destructive &&
                "bg-destructive text-white hover:bg-destructive/90 focus-visible:ring-destructive/20",
            )}
            onClick={() => close(true)}
          >
            {state?.confirmLabel ?? "Confirm"}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );

  return { confirm, dialog };
};

export interface AlertOptions {
  title: ReactNode;
  description?: ReactNode;
  confirmLabel?: string;
}

interface InternalAlertState extends AlertOptions {
  resolve: () => void;
}

interface UseAlertReturn {
  alert: (options: AlertOptions) => Promise<void>;
  dialog: ReactNode;
}

export const useAlert = (): UseAlertReturn => {
  const [state, setState] = useState<InternalAlertState | null>(null);

  const alert = useCallback(
    (options: AlertOptions): Promise<void> =>
      new Promise<void>((resolve) => {
        setState({ ...options, resolve });
      }),
    [],
  );

  const close = (): void => {
    if (state) {
      state.resolve();
      setState(null);
    }
  };

  const dialog = (
    <AlertDialog
      open={state !== null}
      onOpenChange={(open) => {
        if (!open) close();
      }}
    >
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{state?.title}</AlertDialogTitle>
          {state?.description !== undefined && (
            <AlertDialogDescription asChild>
              <div>{state.description}</div>
            </AlertDialogDescription>
          )}
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogAction onClick={() => close()}>
            {state?.confirmLabel ?? "OK"}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );

  return { alert, dialog };
};
