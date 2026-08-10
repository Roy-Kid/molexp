/**
 * UI permission surface — mirrors server role policy
 * (admin / operator write; viewer read-only; users.manage admin-only).
 *
 * When auth is **off**, every capability is allowed (today's open-server DX).
 */

import type { AuthRole, AuthUser } from "./types";

export interface Permissions {
  /** Server process has auth enabled. */
  authEnabled: boolean;
  role: AuthRole | null;
  username: string | null;
  /** Create/delete entities, run lifecycle, settings mutations (non-user). */
  canWrite: boolean;
  /** Settings → Users + operator config credentials. */
  canManageUsers: boolean;
  /**
   * Why write is denied, or null when allowed.
   * Use as button `deniedReason` / menu `title` so hover explains the 403.
   */
  writeDeniedReason: string | null;
  usersDeniedReason: string | null;
}

const WRITE_ROLES = new Set<AuthRole>(["admin", "operator"]);

export function permissionsFromAuth(input: {
  enabled: boolean;
  user: AuthUser | null;
}): Permissions {
  const { enabled, user } = input;
  if (!enabled) {
    return {
      authEnabled: false,
      role: null,
      username: null,
      canWrite: true,
      canManageUsers: true,
      writeDeniedReason: null,
      usersDeniedReason: null,
    };
  }

  const role = user?.role ?? null;
  const username = user?.username ?? null;
  const canWrite = role !== null && WRITE_ROLES.has(role);
  const canManageUsers = role === "admin";

  let writeDeniedReason: string | null = null;
  if (!user) {
    writeDeniedReason = "Sign in required to make changes.";
  } else if (!canWrite) {
    writeDeniedReason =
      role === "viewer"
        ? "Your role is viewer — this server only allows read access. Ask an admin to upgrade you to operator or admin."
        : "You do not have permission to make changes.";
  }

  let usersDeniedReason: string | null = null;
  if (!user) {
    usersDeniedReason = "Sign in required.";
  } else if (!canManageUsers) {
    usersDeniedReason =
      "Only admins can manage users. Your role is " +
      `${role ?? "unknown"} — ask an admin if you need access.`;
  }

  return {
    authEnabled: true,
    role,
    username,
    canWrite,
    canManageUsers,
    writeDeniedReason,
    usersDeniedReason,
  };
}

/** Gate a run-list / tree action that mutates state. */
export function withWriteGate<T extends { disabled?: boolean; title?: string }>(
  action: T,
  writeDeniedReason: string | null,
): T {
  if (!writeDeniedReason) return action;
  return {
    ...action,
    disabled: true,
    title: writeDeniedReason,
  };
}
