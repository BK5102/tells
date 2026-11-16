# Leak audit, Day 11 (2026-11-14)

> Dated record of the Day 11 audit, kept as written. Later status: the agent tool layer and its denied-label test were built (Days 21 to 25), and the configs were sealed on Day 26 (see `2026-11-16-sealed-test-audit.md`). Check 8 was updated in place when the control was built.


Run with the leak audit prompt in `PROMPT.md` section 3. Reviewer stance: find every way the evaluation could cheat. Each check is PASS, FAIL, WARN (works, with a caveat to fix or watch), or NOT YET (the component does not exist yet). Evidence points to code and tests.

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Does any feature at time t use rows from time t or later? | PASS | `cleanroom.pastonly.assert_past_only` rebuilds features from rows strictly before each probe and requires an exact match: `tests/test_checklist_leaks.py` (synthetic data with same-timestamp ties, the TabFormer fixture, 5 real cards) and `tests/test_checklist_aml.py` (AMLWorld fixture, including a send and a receive in the same minute). A planted bug (windows including their own timestamp) failed two of these tests (Day 7). |
| 2 | Do features read the label? | PASS | `assert_ignores` shuffles labels and requires identical features, in the same test files. The builder also drops `label` before computing (`src/tells/features/checklist.py`, `build_checklist_features`). |
| 3 | Do train and test overlap in time? | PASS | `cleanroom.split.Split` rejects overlapping or too-close windows at config time, and `Split.check` re-verifies on the actual rows; the harness calls it on every run (`src/tells/harness.py`, `split_data`). `packages/cleanroom/tests/test_split.py::test_check_catches_a_leaky_part` smuggles a test row into train and confirms it fails. |
| 4 | Do train and test overlap by account in a way that leaks? | WARN | The same account can be in both, on purpose (methodology M16: banks score existing customers). Features describe behavior, not identity, and no account id reaches the model. Watch item: a per-account target encoding would leak through this path; none exists. |
| 5 | Is the lock enforced in code? | PASS | `run_experiment(part="test")` calls `cleanroom.lock.require_lock` and raises `ScoringLockedError` without a lock for the exact config hash (`tests/test_harness.py::test_test_scoring_refused_before_lock`). Any config change needs a new lock (`packages/cleanroom/tests/test_lock.py::test_any_change_to_the_config_needs_a_new_lock`); an edited lock file is rejected (`test_tampered_lock_file_is_rejected`); every test run is counted (`test_every_use_is_recorded`). |
| 6 | Can the lock be bypassed? | WARN | Python code can call `split_data(..., reveal_test_labels=True)` directly; nothing stops a person editing code. This is acceptable for a human-run project because git history shows such edits. Fix before the agent exists (Day 21 to 25): the agent's tools must go through `run_experiment` only, and a guardrail test must show a direct label read is denied. |
| 7 | Does the canary leaky feature get caught? | PASS | `cleanroom.canary.run_canary` plants a future-value feature and a future-label feature; both were caught on the TabFormer and AMLWorld samples with seeds 0, 1, and 2. Every experiment now runs this on its own data and refuses to report numbers otherwise (`src/tells/harness.py`, `canary_check`). Found during this audit: the first version sampled 8 AMLWorld accounts with no laundering, so the outcome canary could not be caught; the sample now includes positive accounts and the Cleanroom refuses a one-outcome sample. |
| 8 | Does a time-shuffled control destroy the signal? | BUILT (Day 18) | Implemented as a label-permutation control (methodology M37): B retrained on labels shuffled within each day must score like random ranking. Unit test passes on synthetic data; real-data results are recorded in each run's `controls` section. |
| 9 | Are report numbers all present in result files? | NOT YET | The reporter is built on Day 24. Results files already carry every number plus provenance. |
| 10 | Are earlier labels used as features without a delay? | PASS | No feature reads labels (check 2). |
| 11 | Is AMLWorld truncated before its end-of-data tail? | PASS | `src/tells/data/amlworld.py` drops rows at or after 2022-09-11; `tests/test_amlworld.py::test_end_of_data_tail_is_truncated` and the full-data count test (4,522 laundering transfers kept). |
| 12 | Is the test period the one locked in the config? | WARN | The configs exist but are deliberately not locked yet: a lock should cover the full plan, including contestants C and D, which do not exist yet. They will be sealed before the Day 26 grid. Until then, the test period cannot be scored at all (check 5). |
| 13 | Does the Cleanroom import project code or contain domain words? | PASS | `packages/cleanroom/tests/test_neutral.py`; it caught the word "credited" in a docstring on Day 5. |
| 14 | Could exchange rates leak later data into AMLWorld amounts? | PASS | Rates use first-day transfers only (`usd_rates`, `tests/test_amlworld.py::test_rates_use_first_day_only`), and the first day is inside every training window. |

## Fixes, in priority order

1. Build the time-shuffled control early (check 8), because the TabFormer scores are high.
2. Before the agent exists, route every agent tool through `run_experiment` and add a denied-label-read test (check 6).
3. Seal the AMLWorld and TabFormer configs before the Day 26 grid, after contestants C and D exist (check 12).
