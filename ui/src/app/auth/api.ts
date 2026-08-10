/**
 * Auth API facade over the generated OpenAPI client.
 * Cookie session via OpenAPI.WITH_CREDENTIALS (set in AuthContext).
 */

import type { AuthUserPublic } from "@/api/generated/models/AuthUserPublic";
import type { CreateUserRequest } from "@/api/generated/models/CreateUserRequest";
import type { PatchUserRequest } from "@/api/generated/models/PatchUserRequest";
import { AuthService } from "@/api/generated/services/AuthService";
import type { AuthRole, AuthStatus, AuthUser } from "./types";

function mapUser(u: AuthUserPublic): AuthUser {
  return {
    username: u.username,
    role: u.role as AuthRole,
    workspaces: u.workspaces ?? [],
    disabled: u.disabled ?? false,
    created_at: u.created_at,
    updated_at: u.updated_at,
  };
}

function asApiRole(role: AuthRole): CreateUserRequest.role {
  return role as unknown as CreateUserRequest.role;
}

function rethrow(err: unknown): never {
  if (err && typeof err === "object" && "body" in err) {
    const body = (err as { body?: { detail?: string }; message?: string }).body;
    const detail = body && typeof body === "object" && "detail" in body ? body.detail : undefined;
    if (typeof detail === "string") {
      throw new Error(detail);
    }
  }
  if (err instanceof Error) {
    throw err;
  }
  throw new Error(String(err));
}

export async function fetchAuthStatus(): Promise<AuthStatus> {
  try {
    const res = await AuthService.authStatusApiAuthStatusGet();
    return {
      enabled: res.enabled,
      authenticated: res.authenticated,
      user: res.user ? mapUser(res.user) : null,
    };
  } catch (err) {
    rethrow(err);
  }
}

export async function login(username: string, password: string): Promise<AuthUser> {
  try {
    const res = await AuthService.authLoginApiAuthLoginPost({ username, password });
    return mapUser(res);
  } catch (err) {
    rethrow(err);
  }
}

export async function logout(): Promise<void> {
  try {
    await AuthService.authLogoutApiAuthLogoutPost();
  } catch (err) {
    rethrow(err);
  }
}

export async function fetchUsers(): Promise<AuthUser[]> {
  try {
    const res = await AuthService.listUsersApiAuthUsersGet();
    return res.users.map(mapUser);
  } catch (err) {
    rethrow(err);
  }
}

export async function createUser(input: {
  username: string;
  password: string;
  role: AuthRole;
  workspaces: string[];
}): Promise<AuthUser> {
  try {
    const res = await AuthService.createUserApiAuthUsersPost({
      username: input.username,
      password: input.password,
      role: asApiRole(input.role),
      workspaces: input.workspaces,
    });
    return mapUser(res);
  } catch (err) {
    rethrow(err);
  }
}

export async function patchUser(
  username: string,
  patch: { role?: AuthRole; workspaces?: string[]; disabled?: boolean },
): Promise<AuthUser> {
  try {
    const body: PatchUserRequest = {
      role:
        patch.role !== undefined ? (asApiRole(patch.role) as PatchUserRequest["role"]) : undefined,
      workspaces: patch.workspaces,
      disabled: patch.disabled,
    };
    const res = await AuthService.patchUserApiAuthUsersUsernamePatch(username, body);
    return mapUser(res);
  } catch (err) {
    rethrow(err);
  }
}

export async function setUserPassword(username: string, password: string): Promise<AuthUser> {
  try {
    const res = await AuthService.setUserPasswordApiAuthUsersUsernamePasswordPost(username, {
      password,
    });
    return mapUser(res);
  } catch (err) {
    rethrow(err);
  }
}

export async function deleteUser(username: string): Promise<void> {
  try {
    await AuthService.deleteUserApiAuthUsersUsernameDelete(username);
  } catch (err) {
    rethrow(err);
  }
}
