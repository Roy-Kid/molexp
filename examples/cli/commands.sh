#!/usr/bin/env bash
# CLI command tour — molexp workspace, project, experiment, run, asset, config.
#
# Matches ``docs/guide/workspace-architecture.md`` and related CLI docs.
#
# Creates a temporary workspace, runs key CLI commands against it,
# and cleans up. Self-contained — no setup needed.
#
# Run directly::
#
#     bash examples/cli/commands.sh

set -euo pipefail

# ── Temporary workspace ───────────────────────────────────────────────────────
WS=$(mktemp -d 2>/dev/null || mktemp -d -t molexp-cli)
cleanup() { rm -rf "$WS"; }
trap cleanup EXIT

echo "Workspace: $WS"

# ── Init ──────────────────────────────────────────────────────────────────────
molexp init "$WS" 2>&1

# ── Info ──────────────────────────────────────────────────────────────────────
echo ""
echo "── molexp info ────────────────────────────────────────────"
molexp info --workspace "$WS" 2>&1 || true

# ── Project create + list ─────────────────────────────────────────────────────
echo ""
echo "── molexp project create ──────────────────────────────────"
molexp project create --workspace "$WS" "demo" 2>&1 || true

echo ""
echo "── molexp project list ────────────────────────────────────"
molexp project list --workspace "$WS" 2>&1 || true

# ── Experiment create + list ──────────────────────────────────────────────────
echo ""
echo "── molexp experiment create ───────────────────────────────"
molexp experiment create --workspace "$WS" --name "baseline" "demo" 2>&1 || true

echo ""
echo "── molexp experiment list ─────────────────────────────────"
molexp experiment list --workspace "$WS" "demo" 2>&1 || true

# ── Runs ──────────────────────────────────────────────────────────────────────
echo ""
echo "── molexp runs list ───────────────────────────────────────"
molexp runs list --workspace "$WS" "demo" "baseline" 2>&1 || true

# ── Asset list ────────────────────────────────────────────────────────────────
echo ""
echo "── molexp asset list ──────────────────────────────────────"
molexp asset list --workspace "$WS" 2>&1 || true

# ── Final info ────────────────────────────────────────────────────────────────
echo ""
echo "── molexp info (summary) ──────────────────────────────────"
molexp info --workspace "$WS" 2>&1 || true

echo ""
echo "Done — all CLI commands completed."
