export type AuthRole = "admin" | "operator" | "viewer";

export interface AuthUser {
  username: string;
  role: AuthRole;
  workspaces: string[];
  disabled: boolean;
  created_at?: string;
  updated_at?: string;
}

export interface AuthStatus {
  enabled: boolean;
  authenticated: boolean;
  user: AuthUser | null;
}
