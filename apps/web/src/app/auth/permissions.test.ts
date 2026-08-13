import { describe, expect, it } from "@rstest/core";
import { permissionsFromAuth, withWriteGate } from "./permissions";

describe("permissionsFromAuth", () => {
  it("allows everything when auth is off", () => {
    const p = permissionsFromAuth({ enabled: false, user: null });
    expect(p.canWrite).toBe(true);
    expect(p.canManageUsers).toBe(true);
    expect(p.writeDeniedReason).toBeNull();
  });

  it("viewer cannot write", () => {
    const p = permissionsFromAuth({
      enabled: true,
      user: {
        username: "v",
        role: "viewer",
        workspaces: ["*"],
        disabled: false,
      },
    });
    expect(p.canWrite).toBe(false);
    expect(p.writeDeniedReason).toMatch(/viewer/i);
    expect(p.canManageUsers).toBe(false);
  });

  it("operator can write but not manage users", () => {
    const p = permissionsFromAuth({
      enabled: true,
      user: {
        username: "op",
        role: "operator",
        workspaces: ["*"],
        disabled: false,
      },
    });
    expect(p.canWrite).toBe(true);
    expect(p.writeDeniedReason).toBeNull();
    expect(p.canManageUsers).toBe(false);
    expect(p.usersDeniedReason).toMatch(/admin/i);
  });

  it("admin can manage users", () => {
    const p = permissionsFromAuth({
      enabled: true,
      user: {
        username: "a",
        role: "admin",
        workspaces: ["*"],
        disabled: false,
      },
    });
    expect(p.canWrite).toBe(true);
    expect(p.canManageUsers).toBe(true);
    expect(p.usersDeniedReason).toBeNull();
  });
});

describe("withWriteGate", () => {
  it("disables and sets title when denied", () => {
    const gated = withWriteGate({ id: "x", disabled: false, title: "ok" }, "No write access");
    expect(gated.disabled).toBe(true);
    expect(gated.title).toBe("No write access");
  });

  it("leaves action alone when allowed", () => {
    const action = { id: "x", disabled: false, title: "ok" };
    expect(withWriteGate(action, null)).toBe(action);
  });
});
