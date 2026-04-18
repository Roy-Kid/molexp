# molcfg + Profiles: Config Files and the Profile Mechanism

**Status**: Draft · **Author**: @RoyKid · **Date**: 2026-04-13

## 1. Motivation

The old `--dry-run` was a hard-coded first-class execution mode, rooted across the CLI, `RunStatus`, `RunMetadata`, `ExecutionConfig`, `ctx`, UI badges, and more. Further requirements (`--profile smoke`, `--dataset-md17`, speed tests, debug slices) would have turned the CLI into a CMake-style flag jungle.

**Goal**: cover every parameter variant (dataset, epochs, batch_size, whether dry-run skips heavy computation, …) with a single mechanism — named profiles in a config file.

**Profile semantics are opaque to the framework.** The framework only loads configs, selects slices, injects them into `ctx`, and records the profile name on the run metadata. Which fields live in a slice, and how tasks interpret them (including "does this have persistent side effects?"), is entirely up to the user. The framework does not ship semantic fields (no `side_effects`, no `real`, no `persist`).

**Non-goal**: backwards compatibility for `ctx.dry_run` / `RunStatus.DRY_RUN` / `--dry-run` as independent fields. Everything is replaced, no deprecation window.

## 2. Design Principles

1. **Stable CLI.** `molexp run script.py [--config X.yaml] [--profile NAME]` is the only entry point; new variants change the config, not the CLI.
2. **A profile is a named config slice.** The profile name is itself the UI badge label, a `RunMetadata` field, and a resume filter.
3. **The framework is blind to profile contents.** Every field is user data. Task code reads `ctx.config["epochs"]` etc. freely. The framework does not interpret.
4. **Profile names are normalized.** `-` is replaced by `_` (YAML `dry-run` is equivalent to `dry_run`), so Python-identifier contexts and CLI dash/underscore conventions agree.
5. **Format**: YAML (default) or JSON. No TOML (the stdlib module is read-only).
6. **Profiles inherit**: explicit `extends: NAME`; no YAML anchors (JSON can't express them, and they introduce too much magic).

## 3. Schema Draft

### 3.1 Config file

```yaml
# molcfg.yaml
version: 1

# Defaults (inherited by every profile)
defaults:
  dataset: md17
  epochs: 100
  batch_size: 32
  seed_base: 42

profiles:
  dry-run:                # normalized to "dry_run" on load
    extends: defaults
    epochs: 1

  smoke:
    extends: defaults
    epochs: 5
    batch_size: 8

  md22:
    extends: defaults
    dataset: md22

  prod:
    extends: defaults
    # deliberately empty — equivalent to defaults
```

### 3.2 Pydantic models

```python
# src/molexp/config/models.py
class ProfileConfig(Mapping[str, Any]):
    """Immutable merged config for one profile.

    Behaves like a read-only dict of user data; carries a ``name`` attr
    (normalized, "-" → "_"; ``None`` means "no profile, defaults only").
    The framework adds no semantic fields — every key comes from the YAML.
    """
    name: str | None
    # internal: frozen dict

    def __getitem__(self, key: str) -> Any: ...
    def get(self, key: str, default=None) -> Any: ...
    def content_hash(self) -> str: ...  # keyed into RunMetadata.config_hash

class MolCfg(BaseModel, frozen=True):
    version: int = 1
    defaults: dict[str, Any] = Field(default_factory=dict)
    profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def resolve(self, name: str | None) -> ProfileConfig:
        """Merge defaults + (optional) profile; normalize name; freeze."""
```

### 3.3 Context injection

```python
class TaskContext(Generic[StateT, DepsT, InputT]):
    config: ProfileConfig   # new
    # dry_run property removed
```

Task-author migration (any "skip side-effects" check becomes a user-defined config field):

```python
# Before
if ctx.dry_run:
    return mock_result()

# After — user adds a field in YAML, task reads it
# molcfg.yaml:
#   profiles:
#     dry-run:
#       skip_heavy_compute: true
if ctx.config.get("skip_heavy_compute"):
    return mock_result()

# Or a pure parameter variant
if ctx.config["epochs"] < 5:
    ...

# Or branch on profile name (not recommended — couples to naming)
if ctx.config.name == "dry_run":
    ...
```

### 3.4 RunMetadata

```python
class RunMetadata(BaseModel):
    id: str
    status: str = "pending"        # pending/running/succeeded/failed/cancelled
    profile: str | None = None     # new: activated profile name, e.g. "dry-run"
    config_hash: str | None = None # new: content-hash of the resolved ProfileConfig
    # removed: dry_run: bool
    ...
```

### 3.5 RunStatus

```python
class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    # removed: DRY_RUN
```

Profile and status are orthogonal. A run with profile `dry-run` that finishes successfully is still `SUCCEEDED`.

### 3.6 CLI

```
molexp run script.py [--config PATH] [--profile NAME] [--local|--slurm|...]
                     [--resume]

--config PATH    defaults to ./molcfg.yaml; if missing, --profile must be empty or the command errors
--profile NAME   activated profile name; default none (= use defaults only)
--resume         re-run all runs whose profile matches and whose status is not succeeded
```

**Removed**: `--dry-run` flag. Use `--profile dry-run` instead.

### 3.7 UI

- Run list: show a `[{profile}]` badge next to status whenever `profile` is non-empty.
- Color mapping is configurable; defaults: `dry-run` = yellow, `smoke` = cyan, others = dim.
- No more `dry_run` status badge.

### 3.8 Resume semantics

New rule: `--resume --profile X` finds every run where `profile == X and status != "succeeded"` and re-executes it. The old "resume only dry-run-status runs" logic generalizes to "resume any profile".

## 4. Roadmap

**Phase 1** (this change): core replacement

- Introduce `molexp.config` (MolCfg / ProfileConfig / loader)
- Replace `ExecutionConfig` with `ProfileConfig`
- Remove `RunStatus.DRY_RUN` and `RunMetadata.dry_run`
- Switch CLI to `--config` / `--profile`
- Sync server / schemas
- Tests stay green

**Phase 2** (follow-up PR): UI

- New profile badge component
- Configurable badge colors

**Phase 3** (later): docs

- Rewrite the quick-start
- Update the README

## 5. Todo List

See the tracker; the critical path is:

1. Create `src/molexp/config/`: models + loader + resolver
2. Update `workspace/models.py`: `ExecutionConfig` → `ProfileConfig`, `RunMetadata` fields
3. Update `workspace/run.py`: drop the `DRY_RUN` enum; `ctx` exposes `config` instead of `dry_run`
4. Update `workflow/context.py`: replace the `dry_run` property with `config`
5. Update `workflow/_pydantic_graph/state.py + node.py + runtime.py + compiler.py`
6. Update `workflow/spec.py + runtime.py`: `execute()` accepts a profile
7. Update `cli/__init__.py`: `--config` / `--profile`; delete `--dry-run` and its branches
8. Update `server/routes/run.py`: status list
9. Update `monitor.py`: badge logic
10. Migrate existing tests (`tests/`)
11. Add new tests: config loading, profile resolution, inheritance
12. Run full pytest, confirm green
