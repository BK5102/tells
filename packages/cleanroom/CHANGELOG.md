# Changelog

## 0.2.0 (2026-11-17, to be tagged `cleanroom-v0.2`)

New pieces for the sibling projects (forecasting curves, and detecting then explaining faults in time series). Nothing existing was removed or changed in a breaking way; code written for 0.1 runs unchanged.

- New dependency: `scipy>=1.17`, for the t and normal distributions used by `stats.diebold_mariano` and `stats.wilson_interval`.
- `cleanroom.lock`: `seal` takes an optional `data_sha256` and stores it in the lock; `require_lock` takes an optional `data_sha256` and refuses a different one when the lock pinned it. Locks without the field and callers that do not pass it behave as before (not breaking). Merged before this release; listed here because it ships in 0.2.0.
- `cleanroom.forecast` (new): `pinball_loss`, `weighted_quantile_loss` (the WQL used by GluonTS and the Chronos paper: twice the pinball loss summed over items, divided by the sum of absolute outcomes, averaged over quantiles), `interval_coverage` (returns a `Coverage` with a Wilson interval), `interval_score` (Gneiting and Raftery 2007), `relative_error` (a ratio, or the geometric mean of ratios).
- `cleanroom.stats`: `diebold_mariano` (returns `DMResult`) with the Harvey, Leybourne and Newbold small-sample correction and a t distribution with n - 1 degrees of freedom. The long-run variance uses lags up to horizon - 1 with equal weights by default (`kernel="uniform"`, as in the original papers); `kernel="bartlett"` gives Newey-West weights, which are always positive but over-reject in this package's simulation (about 9% at a nominal 5% for horizons 2 and 3). `wilson_interval` for a binomial share.
- `cleanroom.split`: `rolling_origins(start, end, step, horizon, gap, train_start)` returns a list of expanding-window `Split`s with `min_gap = gap`.
- `cleanroom.asof` (new): `join_as_known` returns, for each query time, the latest release of each key whose release time is strictly before the query time (`inclusive=True` allows equal times). Accepts one time, a sequence of times (a snapshot), or a query frame.
- `cleanroom.events` (new): `match_alarms` (returns `AlarmMatch`: detected flag and delay per interval, matched flag per alarm), `caught_within`, `false_alarms_per_unit`, `persistent_alarms` (k consecutive exceedances, one alarm per run), and `threshold_for_alarm_rate` (returns `AlarmThreshold`, the lowest calibration-score threshold whose alarm rate, and every higher threshold's, is at most the target; optional groups).
- `cleanroom.guard` (new): `open_sealed_lock`, `start_sealed_run`, `sealed_run` (context manager), `count_runs`, `SealedRun`, `SealedRunError`. A sealed run needs a matching lock, a clean git tree, and a lock tracked by git, and writes "started" and "finished" records to the lock's uses file.
- `cleanroom.vault` (new): `seal_answer_key`, `load_vault`, `release_answer_key`, `Vault`, `VaultError`. The key is released, as the bytes that were checked, only for a valid lock and vault record that git tracks, a run whose uses file shows it finished, and an unchanged sha256; each release is logged.
- `cleanroom.provenance`: `is_tracked(path)`.

## 0.1.0 (2026-11-14, tag `cleanroom-v0.1`)

- Package skeleton inside the Tells uv workspace.
- `cleanroom.split`: `Window`, `Split` (train, optional valid, test, `min_gap`), `apply`, `check`, `to_dict` / `from_dict`. Errors: `SplitError` (bad config), `LeakError` (data breaks time order or gap).
- `cleanroom.metrics`: `budget_selection` (per-group capacity, ties share slots), `recall_at_budget`, `precision_at_budget`, `average_precision`.
- `cleanroom.calibration`: `brier_score`, `reliability_table` (quantile or uniform bins), `expected_calibration_error`.
- `cleanroom.pastonly`: `assert_past_only` (rebuild from strictly earlier rows and compare), `assert_ignores` (shuffled columns must not matter).
- `cleanroom.provenance`: `config_hash`, `file_sha256`, `git_state`, `package_versions`, `stamp`.
- `cleanroom.lock`: `seal`, `require_lock`, `Lock.record_use`, `Lock.uses`. Lock files are never overwritten; an edited lock is rejected.
- `cleanroom.bootstrap`: `cluster_bootstrap_indices`, `paired_bootstrap` (percentile interval, two-sided p-value with a (count + 1) / (n + 1) floor).
- `cleanroom.stats`: `holm`, `benjamini_hochberg`.
- `cleanroom.canary`: `run_canary` plants a future-value and a future-label feature and raises `CanaryNotCaughtError` if either is missed; refuses one-outcome samples.
