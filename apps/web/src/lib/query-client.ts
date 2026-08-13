import { QueryClient } from "@tanstack/react-query";

/** Shared QueryClient for the SPA (auth, settings, future entity data). */
export function createAppQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 30_000,
        retry: 1,
        refetchOnWindowFocus: false,
      },
    },
  });
}
