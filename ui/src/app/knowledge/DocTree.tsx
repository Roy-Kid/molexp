import {
  Blocks,
  BookOpen,
  FilePlus,
  FlaskConical,
  Folder,
  FolderInput,
  NotebookPen,
  Pencil,
  PlayCircle,
  Plus,
  Trash2,
} from "lucide-react";
import type { ComponentType, JSX } from "react";
import type { TreeNode, TreeNodeAction } from "@/app/panels/TreeView";
import { TreeView } from "@/app/panels/TreeView";
import type { Selection, WorkspaceSnapshot } from "@/app/types";
import { useAlert, useConfirm } from "@/components/ConfirmDialog";
import { usePrompt } from "@/components/PromptDialog";
import { Button } from "@/components/ui/button";
import { buildDocTree, type DocEntityKind, type DocTreeNode } from "./knowledgeDocTree";
import { useKnowledgeDocs } from "./useKnowledgeDocs";

interface DocTreeProps {
  snapshot: WorkspaceSnapshot;
  /** The currently-selected Note's bundle-relative path (drives active row). */
  activeId?: string;
  onSelect: (selection: Selection) => void;
}

const ENTITY_ICON: Record<DocEntityKind, ComponentType<{ className?: string }>> = {
  project: Blocks,
  experiment: FlaskConical,
  run: PlayCircle,
};

const collectExpandIds = (nodes: DocTreeNode[], acc: string[]): string[] => {
  for (const node of nodes) {
    if (node.children.length > 0) {
      acc.push(node.id);
      collectExpandIds(node.children, acc);
    }
  }
  return acc;
};

export const DocTree = ({ snapshot, activeId, onSelect }: DocTreeProps): JSX.Element => {
  const { notes, error, createDoc, renameDoc, moveDoc, deleteDoc } = useKnowledgeDocs();
  const { prompt, dialog: promptDialog } = usePrompt();
  const { confirm, dialog: confirmDialog } = useConfirm();
  const { alert, dialog: alertDialog } = useAlert();

  const tree = buildDocTree(notes.map((n) => ({ relPath: n.relPath, name: n.name })));

  const guard = async (label: string, run: () => Promise<void>): Promise<void> => {
    try {
      await run();
    } catch (err) {
      await alert({
        title: label,
        description: err instanceof Error ? err.message : String(err),
      });
    }
  };

  const handleCreateRoot = async (): Promise<void> => {
    const name = await prompt({
      title: "New document",
      label: "Document name",
      placeholder: "My note",
      confirmLabel: "Create",
    });
    if (!name) return;
    await guard("Failed to create document", () => createDoc(name));
  };

  const handleCreateChild = async (parentPath: string): Promise<void> => {
    const name = await prompt({
      title: "New child document",
      label: "Document name",
      description: parentPath,
      confirmLabel: "Create",
    });
    if (!name) return;
    await guard("Failed to create document", () => createDoc(name, parentPath));
  };

  const handleRename = async (path: string, current: string): Promise<void> => {
    const name = await prompt({
      title: "Rename document",
      label: "New name",
      defaultValue: current,
      confirmLabel: "Rename",
    });
    if (!name || name === current) return;
    await guard("Failed to rename document", () => renameDoc(path, name));
  };

  const handleMove = async (path: string): Promise<void> => {
    const parentPath = await prompt({
      title: "Move document",
      label: "New parent path",
      description: "The bundle-relative path of the parent document.",
      placeholder: "kb/parent-note",
      confirmLabel: "Move",
    });
    if (!parentPath) return;
    await guard("Failed to move document", () => moveDoc(path, parentPath));
  };

  const handleDelete = async (path: string, name: string): Promise<void> => {
    const confirmed = await confirm({
      title: "Delete document?",
      description: (
        <>
          Document <code className="rounded bg-muted px-1 py-0.5 text-xs">{name}</code> and its
          child documents will be permanently removed.
        </>
      ),
      confirmLabel: "Delete",
      destructive: true,
    });
    if (!confirmed) return;
    await guard("Failed to delete document", () => deleteDoc(path));
  };

  const docActions = (node: DocTreeNode): TreeNodeAction[] => {
    if (node.relPath === null) return [];
    const path = node.relPath;
    return [
      {
        id: "new-child",
        label: "New child document",
        icon: FilePlus,
        onSelect: () => void handleCreateChild(path),
      },
      {
        id: "rename",
        label: "Rename",
        icon: Pencil,
        onSelect: () => void handleRename(path, node.name),
      },
      {
        id: "move",
        label: "Move",
        icon: FolderInput,
        onSelect: () => void handleMove(path),
      },
      {
        id: "delete",
        label: "Delete",
        icon: Trash2,
        destructive: true,
        separatorBefore: true,
        onSelect: () => void handleDelete(path, node.name),
      },
    ];
  };

  const entityLabel = (kind: DocEntityKind, id: string): string => {
    if (kind === "project") return snapshot.projects.find((p) => p.id === id)?.name ?? id;
    if (kind === "experiment") return snapshot.experiments.find((e) => e.id === id)?.name ?? id;
    return snapshot.runs.find((r) => r.id === id)?.name ?? id;
  };

  const toTreeNode = (node: DocTreeNode): TreeNode => {
    if (node.kind === "doc" && node.relPath) {
      const path = node.relPath;
      return {
        id: node.id,
        label: node.name,
        icon: NotebookPen,
        iconClassName: "text-muted-foreground",
        onSelect: () => onSelect({ objectType: "knowledge", objectId: path }),
        actions: docActions(node),
        children: node.children.length > 0 ? node.children.map(toTreeNode) : undefined,
      };
    }
    if (node.kind === "group") {
      const isKb = node.entity === undefined;
      const label = node.entity ? entityLabel(node.entity.kind, node.entity.id) : node.name;
      const icon = node.entity ? ENTITY_ICON[node.entity.kind] : BookOpen;
      return {
        id: node.id,
        label,
        icon,
        iconClassName: isKb ? "text-blue-500" : "text-muted-foreground",
        labelClassName: "font-semibold",
        actions: isKb
          ? [
              {
                id: "new-doc",
                label: "New document",
                icon: Plus,
                onSelect: () => void handleCreateRoot(),
              },
            ]
          : undefined,
        children: node.children.map(toTreeNode),
      };
    }
    // Intermediate directory segment (no Note of its own).
    return {
      id: node.id,
      label: node.name,
      icon: Folder,
      iconClassName: "text-muted-foreground",
      children: node.children.map(toTreeNode),
    };
  };

  const nodes = tree.map(toTreeNode);
  const expandPath = collectExpandIds(tree, []);

  return (
    <div className="space-y-2">
      <div className="flex justify-end px-1">
        <Button
          variant="ghost"
          size="sm"
          className="h-7 gap-1.5 text-xs"
          onClick={() => void handleCreateRoot()}
        >
          <Plus className="h-3.5 w-3.5" /> New doc
        </Button>
      </div>
      {error && <p className="px-2 text-xs text-destructive">{error}</p>}
      <TreeView
        nodes={nodes}
        activeId={activeId}
        expandPath={expandPath}
        emptyTitle="No documents yet"
        emptyDescription="Create a document to start your knowledge base."
        emptyIcon={<BookOpen className="h-8 w-8" />}
      />
      {promptDialog}
      {confirmDialog}
      {alertDialog}
    </div>
  );
};
