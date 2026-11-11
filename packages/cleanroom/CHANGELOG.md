# Changelog

## Unreleased (0.1.0.dev0)

- Package skeleton inside the Tells uv workspace.
- `cleanroom.split`: `Window`, `Split` (train, optional valid, test, `min_gap`), `apply`, `check`, `to_dict` / `from_dict`. Errors: `SplitError` (bad config), `LeakError` (data breaks time order or gap).
- `cleanroom.metrics`: `budget_selection` (per-group capacity, ties share slots), `recall_at_budget`, `precision_at_budget`, `average_precision`.
- `cleanroom.calibration`: `brier_score`, `reliability_table` (quantile or uniform bins), `expected_calibration_error`.
- `cleanroom.pastonly`: `assert_past_only` (rebuild from strictly earlier rows and compare), `assert_ignores` (shuffled columns must not matter).
- `cleanroom.provenance`: `config_hash`, `file_sha256`, `git_state`, `package_versions`, `stamp`.
