import { buildRegistryKey, registerRendererContribution } from "@/app/registry";
import type { FileKind } from "@/app/types";
import type { UiPluginModule } from "@/plugins/types";
import { TextEditor } from "./TextEditor";

/**
 * Internal `editor` UI plugin — peer of `molvis`, `molq`, `metrics`,
 * `tensorboard`. Owns the `panelKind:"editor"` renderer slot for workspace
 * files, which used to be registered by `core` (`registerDefaultRenderers`).
 *
 * Extension point: the editor registers each contribution under a stable,
 * non-colliding id (`editor:default:<registryKey>`) with an explicit low
 * `priority` (0). The renderer registry resolves the highest-`priority`
 * contribution for a key, so an alternative editor can override the default
 * simply by registering a `panelKind:"editor"` contribution with a *different*
 * id and a higher `priority` — no change to the resolver, no duplicate-id
 * throw. The registry stays the sole extension point.
 *
 * Preview-host contract: `TextEditor` (this plugin's component) is the host of
 * the `editor` panel slot. It renders an Edit/Preview tab pair and, in the
 * Preview tab, delegates to whichever `FilePreviewPlugin` the
 * `filePreviewPluginRegistry` resolves. Preview *content* is supplied by other
 * plugins (core, molvis, …); this plugin only owns the hosting surface.
 */
const EDITOR_FILE_KINDS: readonly FileKind[] = [
  "yaml",
  "json",
  "python",
  "markdown",
  "text",
  "unknown",
];

const editorPlugin: UiPluginModule = {
  id: "editor",
  register: () => {
    for (const fileKind of EDITOR_FILE_KINDS) {
      const key = {
        objectType: "workspace-file" as const,
        fileKind,
        contentType: "text" as const,
        panelKind: "editor" as const,
      };
      registerRendererContribution({
        id: `editor:default:${buildRegistryKey(key)}`,
        priority: 0,
        key,
        title: "Text Editor",
        panelSlot: "center",
        Component: TextEditor,
      });
    }
  },
};

export default editorPlugin;
