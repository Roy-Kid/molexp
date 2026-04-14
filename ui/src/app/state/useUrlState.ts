import { useCallback, useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import type {
  AgentSelection,
  FileKind,
  LeftPanelView,
  Selection,
  SemanticObjectType,
  WorkflowSelection,
  WorkspaceFileSelection,
} from "@/app/types";

const DEFAULT_VIEW: LeftPanelView = "projects";

const viewParamValues: LeftPanelView[] = [
  "workspace",
  "projects",
  "asset",
  "workflow",
  "agent",
];

const objectTypeValues: SemanticObjectType[] = [
  "project",
  "experiment",
  "run",
  "asset",
  "workflow",
  "workspace-file",
  "agent",
];

const fileKindValues: FileKind[] = [
  "yaml",
  "json",
  "python",
  "markdown",
  "text",
  "image",
  "unknown",
];

const isLeftPanelView = (value: string): value is LeftPanelView => {
  return viewParamValues.includes(value as LeftPanelView);
};

const isSemanticObjectType = (value: string): value is SemanticObjectType => {
  return objectTypeValues.includes(value as SemanticObjectType);
};

const isFileKind = (value: string): value is FileKind => {
  return fileKindValues.includes(value as FileKind);
};

export interface UrlState {
  leftPanelView: LeftPanelView;
  selection: Selection | null;
  setLeftPanelView: (view: LeftPanelView) => void;
  setSelection: (selection: Selection | null) => void;
}

const buildSelection = (
  objectType: SemanticObjectType | null,
  objectId: string | null,
  filePath: string | null,
  fileKind: FileKind | null,
  workflowId: string | null,
): Selection | null => {
  if (!objectType || !objectId) {
    return null;
  }

  if (objectType === "workspace-file") {
    const workspaceFile: WorkspaceFileSelection = {
      objectType: "workspace-file",
      objectId,
      filePath: filePath ?? objectId,
      fileKind: fileKind ?? "unknown",
    };
    return workspaceFile;
  }

  if (objectType === "workflow") {
    const workflowSelection: WorkflowSelection = {
      objectType: "workflow",
      objectId,
      workflowId: workflowId ?? objectId,
    };
    return workflowSelection;
  }

  if (objectType === "agent") {
    const agentSelection: AgentSelection = { objectType: "agent", objectId };
    return agentSelection;
  }

  return {
    objectType,
    objectId,
  };
};

export const useUrlState = (): UrlState => {
  const [searchParams, setSearchParams] = useSearchParams();

  const leftPanelView = useMemo<LeftPanelView>(() => {
    const viewParam = searchParams.get("view");
    return viewParam && isLeftPanelView(viewParam) ? viewParam : DEFAULT_VIEW;
  }, [searchParams]);

  const selection = useMemo<Selection | null>(() => {
    const typeParam = searchParams.get("type");
    const idParam = searchParams.get("id");
    const filePath = searchParams.get("file");
    const fileKindParam = searchParams.get("fileKind");
    const workflowIdParam = searchParams.get("workflowId");

    const objectType = typeParam && isSemanticObjectType(typeParam) ? typeParam : null;
    const fileKind = fileKindParam && isFileKind(fileKindParam) ? fileKindParam : null;

    return buildSelection(objectType, idParam, filePath, fileKind, workflowIdParam);
  }, [searchParams]);

  const setLeftPanelView = useCallback(
    (view: LeftPanelView): void => {
      const params = new URLSearchParams(searchParams);
      params.set("view", view);
      setSearchParams(params, { replace: true });
    },
    [searchParams, setSearchParams],
  );

  const setSelection = useCallback(
    (nextSelection: Selection | null): void => {
      const params = new URLSearchParams(searchParams);

      if (!nextSelection) {
        params.delete("type");
        params.delete("id");
        params.delete("file");
        params.delete("fileKind");
        params.delete("workflowId");
        setSearchParams(params, { replace: true });
        return;
      }

      params.set("type", nextSelection.objectType);
      params.set("id", nextSelection.objectId);

      if (nextSelection.objectType === "workspace-file") {
        params.set("file", nextSelection.filePath);
        params.set("fileKind", nextSelection.fileKind);
      } else {
        params.delete("file");
        params.delete("fileKind");
      }

      if (nextSelection.objectType === "workflow") {
        params.set("workflowId", nextSelection.workflowId);
      } else {
        params.delete("workflowId");
      }

      setSearchParams(params, { replace: true });
    },
    [searchParams, setSearchParams],
  );

  return { leftPanelView, selection, setLeftPanelView, setSelection };
};
