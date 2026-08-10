# Regression — plan-step-audit-05-ui

Public UI checks (run from repo root):

```bash
npm run test:ui -- ReviewSurface ApprovalsInbox
```

Expected:

- `ReviewSurface` field-type matrix + `collectFieldValues` green
- `ApprovalsInbox.contract` asserts Approve / Reject / Revise + `fieldValues` on revise
- Generated client exposes `action` / `fieldValues` / `formDocument` (via prior `npm run generate:api`)

Also:

```bash
npm run typecheck:ui
```
