import type { ReactNode } from "react";
import { useCallback, useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export interface PromptOptions {
  title: ReactNode;
  description?: ReactNode;
  placeholder?: string;
  defaultValue?: string;
  label?: string;
  confirmLabel?: string;
  cancelLabel?: string;
}

interface InternalState extends PromptOptions {
  resolve: (value: string | null) => void;
}

interface UsePromptReturn {
  prompt: (options: PromptOptions) => Promise<string | null>;
  dialog: ReactNode;
}

export const usePrompt = (): UsePromptReturn => {
  const [state, setState] = useState<InternalState | null>(null);
  const [value, setValue] = useState("");

  useEffect(() => {
    if (state) setValue(state.defaultValue ?? "");
  }, [state]);

  const prompt = useCallback(
    (options: PromptOptions): Promise<string | null> =>
      new Promise<string | null>((resolve) => {
        setState({ ...options, resolve });
      }),
    [],
  );

  const close = (result: string | null): void => {
    if (state) {
      state.resolve(result);
      setState(null);
    }
  };

  const handleSubmit = (event: React.FormEvent): void => {
    event.preventDefault();
    const trimmed = value.trim();
    close(trimmed.length === 0 ? null : trimmed);
  };

  const dialog = (
    <Dialog
      open={state !== null}
      onOpenChange={(open) => {
        if (!open) close(null);
      }}
    >
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{state?.title}</DialogTitle>
          {state?.description !== undefined && (
            <DialogDescription asChild>
              <div>{state.description}</div>
            </DialogDescription>
          )}
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            {state?.label && <Label htmlFor="prompt-input">{state.label}</Label>}
            <Input
              id="prompt-input"
              autoFocus
              value={value}
              placeholder={state?.placeholder}
              onChange={(event) => setValue(event.target.value)}
            />
          </div>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => close(null)}>
              {state?.cancelLabel ?? "Cancel"}
            </Button>
            <Button type="submit">{state?.confirmLabel ?? "OK"}</Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );

  return { prompt, dialog };
};
