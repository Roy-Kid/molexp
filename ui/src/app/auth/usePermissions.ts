import { useMemo } from "react";
import { useAuth } from "./AuthContext";
import { type Permissions, permissionsFromAuth } from "./permissions";

/** Live permissions for the signed-in user (or open server when auth is off). */
export function usePermissions(): Permissions {
  const { enabled, user } = useAuth();
  return useMemo(() => permissionsFromAuth({ enabled, user }), [enabled, user]);
}
