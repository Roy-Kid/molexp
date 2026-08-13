/**
 * Auth surface backed by TanStack Query — status is a query; login/logout
 * are mutations that invalidate the status key. No hand-rolled fetch state.
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { createContext, type ReactNode, useCallback, useContext, useMemo } from "react";
import { OpenAPI } from "@/api/generated/core/OpenAPI";
import * as authApi from "./api";
import { authKeys } from "./keys";
import type { AuthStatus, AuthUser } from "./types";

// Cookie sessions must be sent on every generated client call + EventSource.
OpenAPI.WITH_CREDENTIALS = true;
OpenAPI.CREDENTIALS = "include";

const OFF_STATUS: AuthStatus = {
  enabled: false,
  authenticated: false,
  user: null,
};

interface AuthContextValue {
  status: AuthStatus | null;
  loading: boolean;
  error: string | null;
  user: AuthUser | null;
  enabled: boolean;
  authenticated: boolean;
  refresh: () => Promise<void>;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }): JSX.Element {
  const queryClient = useQueryClient();

  const statusQuery = useQuery({
    queryKey: authKeys.status(),
    queryFn: authApi.fetchAuthStatus,
    // Backend without auth routes (old) → treat as auth off so mock/dev still boots.
    retry: false,
  });

  const status: AuthStatus = statusQuery.isError ? OFF_STATUS : (statusQuery.data ?? OFF_STATUS);

  const loginMutation = useMutation({
    mutationFn: ({ username, password }: { username: string; password: string }) =>
      authApi.login(username, password),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: authKeys.status() });
    },
  });

  const logoutMutation = useMutation({
    mutationFn: authApi.logout,
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: authKeys.all });
    },
  });

  const refresh = useCallback(async (): Promise<void> => {
    await queryClient.invalidateQueries({ queryKey: authKeys.status() });
  }, [queryClient]);

  const login = useCallback(
    async (username: string, password: string): Promise<void> => {
      await loginMutation.mutateAsync({ username, password });
    },
    [loginMutation],
  );

  const logout = useCallback(async (): Promise<void> => {
    await logoutMutation.mutateAsync();
  }, [logoutMutation]);

  const value = useMemo<AuthContextValue>(
    () => ({
      status,
      loading: statusQuery.isLoading,
      error:
        statusQuery.isError && statusQuery.error instanceof Error
          ? statusQuery.error.message
          : loginMutation.error instanceof Error
            ? loginMutation.error.message
            : null,
      user: status.user,
      enabled: status.enabled,
      authenticated: status.authenticated,
      refresh,
      login,
      logout,
    }),
    [
      status,
      statusQuery.isLoading,
      statusQuery.isError,
      statusQuery.error,
      loginMutation.error,
      refresh,
      login,
      logout,
    ],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return ctx;
}
