/** TanStack Query keys for auth. */
export const authKeys = {
  all: ["auth"] as const,
  status: () => [...authKeys.all, "status"] as const,
  users: () => [...authKeys.all, "users"] as const,
};
