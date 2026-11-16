# Leak audit #3: the sealed test (2026-11-16)

Checks that the AMLWorld test result was produced the way the plan says: sealed first, scored once, nothing changed in between. Evidence comes from git history and the run files.

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Was the plan sealed before the test was scored? | PASS | Lock `f6ff26be` sealed at 19:06:09 UTC; committed in `4a21200` at 19:06:17 UTC; the test run `20261116T190641Z_amlworld_test_f6ff26be_ac67fd` started at 19:06:41 UTC (its id is its start time). |
| 2 | Is the scored config exactly the sealed one? | PASS | The run's config hash `f6ff26be...` equals the lock's; the lock's stored config re-hashes to the same value. It is also the config hash of both validation runs, including the first one with unstandardized PCA: the lock covers the config file, not the code. The standardization step is pinned by the run's git commit (check 5). |
| 3 | Did configs or locks change after sealing? | PASS | `git diff 4a21200 -- configs locks/*.lock.json` is empty. |
| 4 | How many times was the test period scored? | PASS | Once: `locks/f6ff26be...uses.jsonl` has one entry, and `artifacts/runs/index.jsonl` lists one test run among 15 runs. |
| 5 | Which code scored it? | PASS | The run is stamped with commit `4a21200` and no uncommitted changes; that commit only added the lock files on top of `e77fffe`, the commit recorded inside both locks. |
| 6 | Did the run's own leak checks pass? | PASS | Canary passed; the label-permutation control scored like random ranking (+0.1 points, interval -0.2 to +0.4). |
| 7 | Were later analyses kept separate from the result? | PASS (design); output pending | `scripts/analyze_test_run.py` reads the saved case files only, and its output (`docs/analysis/amlworld-test-descriptive.md`) is labeled descriptive and not pre-registered. |
| 8 | Is the TabFormer plan still untouched? | PASS (in progress) | Lock `701e5f22` sealed in the same commit, before any full-data TSFM run on TabFormer; its validation and test runs are running on the sealed config. |

No failures. The AMLWorld test result in `docs/findings.md` stands as produced.
