import { http, HttpResponse } from "msw";

/**
 * Mock auth is always off so offline ``npm run dev:web`` boots without a login wall.
 * Real auth is exercised against ``molexp serve --auth``.
 */
export const authHandlers = [
  http.get("/api/auth/status", () =>
    HttpResponse.json({
      enabled: false,
      authenticated: false,
      user: null,
    }),
  ),
  http.get("/api/auth/users", () => HttpResponse.json({ users: [] })),
];
