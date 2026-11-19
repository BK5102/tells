#!/usr/bin/env bash
# Short showcase for a terminal recording: tests, a small run, its report, and the
# sealed v0.2 headline. Run from the repo root after the steps in docs/runbook.md:
#   bash scripts/demo.sh           # reuses the newest small run if there is one
#   bash scripts/demo.sh --fresh   # runs configs/small.yaml first (several minutes on a laptop)
# Needs the prepared TabFormer sample (uv run tells prepare tabformer) and the tsfm extra.
set -euo pipefail

fresh=false
[ "${1:-}" = "--fresh" ] && fresh=true

step() { printf '\n== %s ==\n' "$1"; }

step "1. The honest-exam toolkit's own tests (Cleanroom)"
uv run pytest -q packages/cleanroom/tests

step "2. Leak and lock tests for Tells"
uv run pytest -q tests/test_checklist_leaks.py tests/test_harness.py

step "3. A small experiment: 300 TabFormer cards, validation year 2016"
if $fresh || ! ls -d artifacts/runs/*_small_valid_* >/dev/null 2>&1; then
  uv run tells run configs/small.yaml | tail -n 1
else
  echo "reusing the newest run; pass --fresh to run configs/small.yaml again"
fi

step "4. Its report, built from the run files (no number typed by hand)"
latest=$(ls -d artifacts/runs/*_small_valid_* | sort | tail -n 1)
uv run tells report "$latest" --no-history | head -n 30

sealed=artifacts/runs/20261117T235515Z_amlworld_li_v2_test_cf534a03_0f604f
if [ -d "$sealed" ]; then
  step "5. The sealed v0.2 test (scored once under lock cf534a03)"
  uv run tells report "$sealed" --no-history | head -n 20
fi
