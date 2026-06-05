/**
 * MolExp Workflow Preview — VSCode extension entry.
 *
 * Registers a read-only custom editor for `*workflow.json` files. The editor
 * hosts a webview that mounts the molexp UI's `<WorkflowPreview>` component
 * (bundled from `molexp/ui/src/components/workflow`), so the DAG renders with
 * the exact same flowgram canvas + shadcn chrome as `molexp serve`.
 *
 * Data flow: the extension owns the file text and pushes it to the webview on
 * open and on every edit; the webview is a pure renderer and never writes back.
 */

import * as vscode from "vscode";

const VIEW_TYPE = "molexp.workflowPreview";

export function activate(context: vscode.ExtensionContext): void {
  const provider = new WorkflowPreviewProvider(context);
  context.subscriptions.push(
    vscode.window.registerCustomEditorProvider(VIEW_TYPE, provider, {
      webviewOptions: { retainContextWhenHidden: true },
      supportsMultipleEditorsPerDocument: true,
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("molexp.openWorkflowPreview", () => {
      const uri = vscode.window.activeTextEditor?.document.uri;
      if (!uri) {
        void vscode.window.showInformationMessage("Open a workflow.json file first.");
        return;
      }
      void vscode.commands.executeCommand("vscode.openWith", uri, VIEW_TYPE);
    }),
  );
}

export function deactivate(): void {
  /* no-op */
}

class WorkflowPreviewProvider implements vscode.CustomTextEditorProvider {
  constructor(private readonly context: vscode.ExtensionContext) {}

  resolveCustomTextEditor(
    document: vscode.TextDocument,
    panel: vscode.WebviewPanel,
  ): void {
    const dist = vscode.Uri.joinPath(this.context.extensionUri, "dist");
    panel.webview.options = {
      enableScripts: true,
      localResourceRoots: [dist],
    };
    panel.webview.html = this.html(panel.webview, dist);

    const push = (): void => {
      void panel.webview.postMessage({ type: "update", text: document.getText() });
    };

    // Re-push whenever THIS document changes on disk / in another editor.
    const changeSub = vscode.workspace.onDidChangeTextDocument((e) => {
      if (e.document.uri.toString() === document.uri.toString()) push();
    });
    panel.onDidDispose(() => changeSub.dispose());

    // The webview asks for the initial payload once it has mounted (avoids a
    // race where we post before the listener is attached).
    panel.webview.onDidReceiveMessage((msg: { type?: string }) => {
      if (msg?.type === "ready") push();
    });
  }

  private html(webview: vscode.Webview, dist: vscode.Uri): string {
    const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(dist, "webview.js"));
    const themeUri = webview.asWebviewUri(vscode.Uri.joinPath(dist, "theme.css"));
    const canvasCssUri = webview.asWebviewUri(vscode.Uri.joinPath(dist, "webview.css"));
    const nonce = makeNonce();
    const csp = [
      `default-src 'none'`,
      `img-src ${webview.cspSource} https: data:`,
      `style-src ${webview.cspSource} 'unsafe-inline'`,
      `font-src ${webview.cspSource} data:`,
      `script-src 'nonce-${nonce}'`,
    ].join("; ");
    return `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta http-equiv="Content-Security-Policy" content="${csp}" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <link href="${themeUri}" rel="stylesheet" />
    <link href="${canvasCssUri}" rel="stylesheet" />
    <title>MolExp Workflow Preview</title>
  </head>
  <body>
    <div id="root"></div>
    <script nonce="${nonce}" src="${scriptUri}"></script>
  </body>
</html>`;
  }
}

function makeNonce(): string {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let out = "";
  for (let i = 0; i < 32; i++) out += chars.charAt(Math.floor(Math.random() * chars.length));
  return out;
}
