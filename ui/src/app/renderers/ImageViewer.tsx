import { useEffect, useState } from "react";
import { workspaceApi } from "@/app/state/api";
import type { RendererProps } from "@/app/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

export const ImageViewer = ({ selection }: RendererProps): JSX.Element => {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (selection.objectType !== "workspace-file") {
      return;
    }

    let revoked = false;
    let currentUrl: string | null = null;

    workspaceApi
      .getWorkspaceFileBlob(selection.objectId)
      .then((blob) => {
        if (revoked) {
          return;
        }
        currentUrl = URL.createObjectURL(blob);
        setImageUrl(currentUrl);
        setError(null);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : "Failed to load image");
        setImageUrl(null);
      });

    return () => {
      revoked = true;
      if (currentUrl) {
        URL.revokeObjectURL(currentUrl);
      }
    };
  }, [selection]);

  return (
    <Card className="flex h-full flex-col border-border/60 bg-background">
      <CardHeader className="space-y-2">
        <CardTitle className="text-lg font-semibold">Image Preview</CardTitle>
        <p className="text-sm text-muted-foreground">{selection.objectId}</p>
      </CardHeader>
      <Separator />
      <CardContent className="flex-1 pt-4">
        {error && <div className="text-sm text-destructive">{error}</div>}
        {!error && imageUrl && (
          <div className="flex h-full items-center justify-center">
            <img src={imageUrl} alt={selection.objectId} className="max-h-full max-w-full" />
          </div>
        )}
      </CardContent>
    </Card>
  );
};
