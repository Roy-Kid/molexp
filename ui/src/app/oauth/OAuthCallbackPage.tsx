/**
 * OAuthCallbackPage — destination of the IdP redirect after the user logs in.
 *
 * Lives at ``/oauth-callback`` (matching ``MOLEXP_OAUTH_REDIRECT_URI`` /
 * the redirect URI registered with the IdP via Dynamic Client Registration).
 * On mount, it pulls ``code`` and ``state`` from ``window.location.search``,
 * postMessages them to the opener (the McpServersTab popup launcher),
 * and asks the user to close the window. The opener then POSTs to
 * ``/api/agent/mcp/servers/{name}/oauth/callback`` to finish token exchange.
 *
 * Failure modes shown to the user:
 *   - IdP returned an error (``error`` query param) — render the IdP message
 *   - Window opened directly (no opener) — render plain code so the user
 *     can copy it manually if needed
 */

import { useEffect, useMemo, useState } from "react";

interface CallbackParams {
  code: string | null;
  state: string | null;
  error: string | null;
  errorDescription: string | null;
}

const parseQuery = (search: string): CallbackParams => {
  const params = new URLSearchParams(search);
  return {
    code: params.get("code"),
    state: params.get("state"),
    error: params.get("error"),
    errorDescription: params.get("error_description"),
  };
};

export const OAuthCallbackPage = (): JSX.Element => {
  const params = useMemo(() => parseQuery(window.location.search), []);
  const [posted, setPosted] = useState(false);
  const [noOpener, setNoOpener] = useState(false);

  useEffect(() => {
    if (params.error || !params.code) return;
    if (!window.opener || window.opener.closed) {
      setNoOpener(true);
      return;
    }
    window.opener.postMessage(
      {
        type: "molexp:oauth-callback",
        code: params.code,
        state: params.state,
      },
      window.location.origin,
    );
    setPosted(true);
    // Give the opener a moment to flush, then auto-close.
    const t = window.setTimeout(() => {
      try {
        window.close();
      } catch {
        /* some browsers refuse window.close() unless opened by JS */
      }
    }, 600);
    return () => window.clearTimeout(t);
  }, [params]);

  if (params.error) {
    return (
      <Center>
        <Title>Authorization failed</Title>
        <Body>
          <strong>{params.error}</strong>
          {params.errorDescription && <p>{params.errorDescription}</p>}
        </Body>
      </Center>
    );
  }

  if (!params.code) {
    return (
      <Center>
        <Title>Missing authorization code</Title>
        <Body>The IdP did not return an authorization code in the redirect URL.</Body>
      </Center>
    );
  }

  if (noOpener) {
    return (
      <Center>
        <Title>Authorization received</Title>
        <Body>
          This window was not opened from molexp's settings UI. Copy the code below and paste it
          into the Connect dialog manually.
          <pre className="mt-3 max-w-full overflow-auto rounded border bg-muted/40 p-2 text-xs">
            {params.code}
          </pre>
        </Body>
      </Center>
    );
  }

  return (
    <Center>
      <Title>Authorization received</Title>
      <Body>
        {posted
          ? "You can close this window. molexp is finishing the connection."
          : "Forwarding code to molexp…"}
      </Body>
    </Center>
  );
};

const Center = ({ children }: { children: React.ReactNode }): JSX.Element => (
  <div className="flex min-h-screen items-center justify-center bg-background p-6">
    <div className="max-w-md space-y-3 rounded-lg border bg-card p-6 shadow-sm">{children}</div>
  </div>
);

const Title = ({ children }: { children: React.ReactNode }): JSX.Element => (
  <h1 className="text-lg font-semibold">{children}</h1>
);

const Body = ({ children }: { children: React.ReactNode }): JSX.Element => (
  <div className="text-sm text-muted-foreground">{children}</div>
);
