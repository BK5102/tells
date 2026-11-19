# Changelog

All notable changes to Tells. The shared Cleanroom package has its own changelog in `packages/cleanroom/CHANGELOG.md`.

## Unreleased

- `docs/runbook.md`: the owner's three remaining manual steps (live agent run, Kaggle checksum comparison, terminal recording), one command each.
- `scripts/verify_checksums.py`: compares local data, another copy (such as the official Kaggle file), or a pasted checksum with `data/checksums.sha256`.
- `scripts/demo.sh`: a short showcase for a terminal recording.
- Fixed: `tells run` crashed after saving a run when printing its summary, because YAML split dates are not JSON serializable. The run files were unaffected (the harness already wrote them with dates as text).

## 0.2.0 (2026-11-17, tagged `v0.2`)

A pre-registered follow-up on fresh data: `docs/v2-plan.md`.

- **Plan** fixed on 2026-11-17 before any v0.2 model ran: two variants, `ttm_recent` (TTM summary of the most recent patch, 64 standardized PCA components) and `chronos_bolt` (Chronos-Bolt-tiny, revision `a0e552de83495b5c28c14c71c374f3e33280b340`), each as C (TSFM only) and D (B plus TSFM), Holm-corrected against B, headline unchanged (first-fraud-day catch at 0.5%). Sealed test: AMLWorld LI-Small, truncated at 2022-09-11, same calendar split as v0.1, config `configs/amlworld_li_v2.yaml`.
- **Changed before sealing:** MOMENT-1-small, the first choice for variant 2, was measured at 143 ms per account-day on CPU (about 190 CPU hours for v0.2's data, no GPU) and dropped; a sample-only test would hold too few newly bad accounts to detect a gain. The replacement was chosen by a rule written first (the cheapest TSFM from a different family than TTM with embeddings, under 10 ms per account-day). Development moved to the 20,000-account HI-Small sample. Reasons: `docs/v2-plan.md` "Changes before sealing", methodology M46 to M48.
- **Result** (sealed test, AMLWorld LI-Small, 2022-09-08 to 2022-09-10, 1,549 newly bad accounts; run `20261117T235515Z_amlworld_li_v2_test_cf534a03_0f604f`, test run number 1 under lock `cf534a03`, sealed in commit `b8a1927`): D_chronos_bolt caught 24.0% of newly bad accounts on their first bad day at the 0.5% budget against 20.7% for B, +3.2 points [+1.1, +5.4], Holm p 0.016, which meets the pre-registered win rule. D_ttm_recent caught 21.6%, +0.8 [-1.2, +2.8], Holm p 0.42: no detectable gain. The canary passed and the label-permutation control was consistent with chance (+0.0 points, -0.3 to +0.4). A Bonferroni bound over all four sealed TSFM tests in the project gives 0.032. Caveats: one synthetic dataset; a modest effect (roughly 50 more of 1,549 accounts at the same budget); LI-Small validation pointed the other way (-2.1 [-5.9, +1.4]); D behind B at the 2% and 5% budgets (descriptive); the gain is specific to Chronos-Bolt. Details: methodology M49, L25 to L28.

## 0.1.0 (tagged `v0.1`)

The first version of Tells: an evaluation of whether a time series foundation model (TSFM) helps a bank-style model catch accounts on the first day they go bad, under a fixed daily review budget, with the test plan sealed before scoring.

### Added

- **Cleanroom 0.1.0** (tag `cleanroom-v0.1`), the domain-neutral honest-exam package: time splits with gaps, budget metrics, calibration, past-only checks, provenance, the sealed-envelope lock, account-level paired bootstrap, Holm and Benjamini-Hochberg corrections, and the canary. Details in `packages/cleanroom/CHANGELOG.md`. A test fails the build if the package mentions a domain or imports from `tells`.
- **Data loaders** for IBM TabFormer and AMLWorld HI-Small into one standard schema, with seeded account samples (300 TabFormer cards, 20,000 AMLWorld accounts) and `tells prepare <dataset>`. AMLWorld is truncated before 2022-09-11 to remove an end-of-file artifact, and amounts are converted to US dollars with first-day exchange rates.
- **Account-day cases** (`src/tells/cases.py`): one case per account per active day, scored by its most suspicious transaction, with a flag for each account's first bad day.
- **Scoring harness** (`tells run <config>`) with budgets of 0.1%, 0.5%, 1%, 2%, and 5% of each day's cases, the first-fraud-day catch rate as the headline (0.5% on AMLWorld, 0.1% on TabFormer), recall and precision at every budget, case-level and transaction-level PR-AUC, and calibration against a base-rate Brier score.
- **Past-only checklist features**: 34 red-flag features, including money-movement flags (money in versus out, new counterparties) so the bank-style model is a fair anti-money-laundering baseline. Cached by data checksum plus feature-code checksum.
- **Four contestants**: A, rank by amount; B, checklist features into LightGBM with fixed, untuned settings; C, TSFM columns only; D, B's columns plus TSFM columns with B's exact settings.
- **TSFM adapters** for IBM TTM r2 (0.8M parameters, pinned to commit d6a79570) and Amazon Chronos-2 (119.5M parameters, pinned to commit 29ec3766), plus a fake adapter so tests run without PyTorch. Account histories become 256-event sequences cut at midnight; embeddings are cached in resumable chunks, averaged over patches, standardized, and reduced to 16 PCA components fitted on training days only. Two surprise scores (amount and timing against the forecast next event).
- **Controls and explanations**: a canary on every run, a label-permutation control on every run, LightGBM SHAP attributions with the TSFM share of contestant D's attribution, and a split by history length (fewer than 16 prior events versus 16 or more) with the training versus evaluation history mix reported per run.
- **Sealed test plans**: `tells lock <config>` seals a config by hash; test scoring refuses without a matching lock and every test run is counted. Both plans (AMLWorld lock `f6ff26be`, TabFormer lock `701e5f22`) were committed in `4a21200` before any test score.
- **Provenance and run index**: every run stores config hash, git commit and dirty flag, data checksum, library versions, and fit and scoring time, and is appended to `artifacts/runs/index.jsonl`.
- **Report generator** (`tells report <run>`) that builds every table from run files, so no number is typed by hand.
- **Agent** (`tells agent "<question>"`): Claude Opus 5.5 in a loop limited to three tools (`run_experiment`, `load_results`, `plot`) and 10 tool calls, a deterministic Critic that checks every claim against the run files and bootstrap intervals, and a Reporter that labels its output a draft for human review. Tested with a hand-written cassette; a live run needs an API key.
- **Docs**: `docs/methodology.md` (every choice with its reason and limitation), `docs/decisions.md`, `docs/findings.md`, `docs/data-fit.md`, `docs/tsfm-notes.md`, `docs/explain-the-project.md`, and leak audits in `docs/audits/`.

### Results

#### Sealed tests (the results that count)

Across both sealed tests, adding TTM to the bank-style model gave no detectable gain at the pre-registered budget: AMLWorld +0.1 points [-1.6, +1.9] at 0.5%, TabFormer +2.0 points [-0.6, +5.0] at 0.1%.

AMLWorld HI-Small, test period 2022-09-08 to 2022-09-10, 1,798 newly bad accounts. Run `20261116T190641Z_amlworld_test_f6ff26be_ac67fd`, test run number 1 under lock `f6ff26be`, sealed in commit `4a21200` before scoring.

| Contestant | First-fraud-day catch at 0.5% | Difference vs B [95% interval] | Holm p |
|---|---|---|---|
| A. Lazy baseline | 1.6% | -32.8 [-35.0, -30.7] | 0.006 |
| B. Checklist + LightGBM | 34.4% | reference | |
| C. TTM only | 1.0% | -33.4 [-35.6, -31.2] | 0.006 |
| **D. B + TTM** | **34.5%** | **+0.1 [-1.6, +1.9]** | **0.94** |

Adding TTM summaries of account histories did not detectably change how many newly bad accounts were caught on their first bad day at the pre-registered budget; the interval rules out a gain larger than about 2 points. The canary passed, and the label-permutation control scored like random ranking (+0.1 points, -0.2 to +0.4).

Descriptive only, not pre-registered: D was ahead at the tightest budget (13.6% against 11.0% at 0.1%) and on transaction-level PR-AUC (0.398 against 0.321), and behind at loose budgets (58.8% against 63.1% at 2%, 77.3% against 79.5% at 5%). Results were the same for short-history and long-history accounts.

IBM TabFormer (long-history test), test period 2017-04-01 to 2019-12-31, 243 newly bad cards. Run `20261117T045139Z_tabformer_test_701e5f22_2e74f8`, test run number 1 under lock `701e5f22`, sealed in commit `4a21200` before scoring.

| Contestant | First-fraud-day catch at 0.1% | Difference vs B [95% interval] | Holm p |
|---|---|---|---|
| A. Lazy baseline | 1.4% | -9.8 [-13.9, -5.7] | 0.006 |
| B. Checklist + LightGBM | 11.2% | reference | |
| C. TTM only | 2.2% | -9.0 [-12.9, -5.3] | 0.006 |
| **D. B + TTM** | **13.2%** | **+2.0 [-0.6, +5.0]** | **0.16** |

No clear difference at the pre-registered budget. The canary passed, and the label-permutation control was consistent with chance (+0.4 points, -0.1 to +1.4). About 99% of account-days in both training and test had long histories (16 or more earlier events).

Descriptive only, not pre-registered: D was ahead at every budget (33.9% against 29.9% at 0.5%, 47.7% against 43.6% at 1%, 59.9% against 57.2% at 2%, 77.0% against 72.4% at 5%) and on transaction-level PR-AUC (0.114 against 0.101).

#### Validation (before sealing; not findings)

These numbers guided development and are kept on record. One setting (standardizing embeddings before PCA) was chosen after seeing the first validation run, which is why only the sealed test can confirm anything.

| | AMLWorld, full data | AMLWorld, 20,000-account sample | TabFormer, 300-card sample |
|---|---|---|---|
| Validation period | 2022-09-06 | 2022-09-06 | 2016 |
| Budget / newly bad accounts | 0.5% / 672 | 0.5% / 23 | 0.1% / 29 |
| B. Checklist + LightGBM | 31.5% | 39.1% | 9.5% |
| D. B + TTM | 37.2%: +5.7 [+2.3, +9.1], p 0.006 | 34.8%: -4.3 [-23.6, +11.8] | 8.7%: -0.8 [-2.2, 0.0], p 0.31 |
| D. B + Chronos-2 | not run (needs a GPU) | 43.5%: +4.3 [-10.5, +18.2] | not run |

TabFormer on full data, validation 2016, 556 newly bad cards at 0.1% (run `20261116T193405Z_tabformer_valid_701e5f22_d9a420`, same config as the sealed test): A 0.7%, B 28.4%, C 4.2%, D 31.0%; D minus B +2.5 points [-1.1, +6.0], Holm p 0.16. Its label-permutation control was not consistent with chance (see Known limitations).

The first TTM variant on full AMLWorld (unstandardized PCA, run `20261115T031830Z_amlworld_valid_f6ff26be_96919d`) made D worse: 19.9% against 31.5%, -11.6 points [-15.2, -8.0]. The validation gain of +5.7 points came from the second variant and did not hold on the sealed test, which had 2.7 times as many newly bad accounts.

### Fixed

Problems found and fixed during the project, each of which would otherwise have produced a misleading result:

- **AMLWorld end-of-file artifact.** The last days of the file are about 60% laundering against 0.1% before, so any model could score well by learning "late date means laundering". Everything from 2022-09-11 on is dropped (1,108 transfers, including 657 of 5,179 laundering transfers).
- **A dataset that could not test the idea.** IEEE-CIS was set aside after measurement: the median reconstructed account appears once, and 66% of later transactions on a fraud account are labeled fraud, a sign labels were copied forward.
- **A metric with no room to move.** On TabFormer the bank-style model caught about 96% of fraud cases at a 1% budget. A 0.1% budget was added, the first-fraud-day catch became the headline, and AMLWorld became the main test.
- **A headline that counted old news.** The first first-fraud-day metric counted accounts already known to be bad (19% of AMLWorld and 46% of TabFormer positives on validation). It now counts only new catches.
- **A biased bootstrap.** Re-ranking each resample let duplicated accounts compete for the same review slots, and one interval did not contain its own estimate. The queue is now formed once on real data, and every result reports the bootstrap mean so this bias would be visible.
- **A blind canary.** The first AMLWorld canary sample had no laundering accounts, so a label leak could not be detected. Samples now always include positive accounts, and the Cleanroom refuses one-outcome samples.
- **A first TSFM result caused by two problems, not the model.** PCA on raw embeddings was dominated by a few numbers on a huge scale (spreads up to 479 against a median of 7 took 97% of the variance), and AMLWorld training days hold almost only new accounts (9.3% with 16 or more earlier events) while validation is about half established (48.4%). Embeddings are now standardized before PCA, every run reports the history mix, and the first run stays on record as a tried variant.
- **Code-review fixes** (merge `2d6e929`): cached features are validated before reuse; `tells report` never writes the feature cache; test scoring is stricter (it requires a clean checkout and a tracked lock file, and records when each test run started and finished); the agent cannot request the test period unless run with `--allow-test`; Critic fixes; the canary chooses its accounts from before the test window; and the agent loop catches tool errors instead of stopping. The test count is now 187.
- **Same-second run folders.** Two runs starting in the same second shared a folder on fast CI machines (`fa5ea99`). Run folders now get a unique suffix and still refuse to overwrite.

### Known limitations

The full register is in `docs/methodology.md` under "Limitations register". The ten that matter most:

1. **Synthetic data only, from two IBM generators** (L1, L2). Nothing here shows how a model performs at a real bank, and generator quirks may drive results.
2. **AMLWorld has ten usable days** (L3). Gaps are one day, validation is a single day, and intervals are wide.
3. **AMLWorld training holds almost only new accounts** (L18). Account-days with 16 or more earlier events are 9.3% of training and 62.7% of the test period, so TSFM features are learned on new accounts and applied to established ones. No split choice removes this.
4. **Chronos-2 ran only on a 20,000-account sample** (L15, L16). Full-data Chronos-2 needs a GPU, so the sealed test covers TTM only.
5. **No graph features in contestant B** (L7). B is weaker than the best published AMLWorld baselines, though its money-movement red flags keep it from being a strawman.
6. **The AMLWorld copy is a Hugging Face mirror** (L4). Its counts match the paper; the checksum comparison with the official Kaggle file is still to do (one command in `docs/runbook.md`).
7. **The gaps are assumptions** (L6). 90 days for cards and 1 day for laundering stand in for label delay; real delays differ.
8. **The lock does not block code edits** (L13). Git history is what makes any edit between sealing and scoring visible.
9. **A label-permutation control failed on TabFormer validation.** The bank-style model trained on labels shuffled within each day caught 5.9% of newly bad cards against 0.1% for random ranking (+5.8 points [+4.0, +7.8]). The same control passed on the TabFormer sealed test. A follow-up that shuffles across all days (`docs/analysis/tabformer-shuffle-scope-control.md`, generated after the sealed runs) cut the excess at 0.1% to +1.5 [+0.6, +2.6] and +0.9 [+0.3, +1.6] (two seeds). That model holds no label information, so the remainder is not a leak: busier account-days get higher maximum scores even under noise, and a tree fitted to noise still flags unusual transactions. A constant score is too weak a stand-in for chance here (methodology M42, L24).
10. **The bank-style model is never retrained.** On TabFormer, B's catch at 0.5% fell from 90.1% on validation (2016) to 29.9% on the 2017 to 2019 test period, consistent with a model trained on data up to October 2015 going stale. Descriptive only; a rolling retrain is the next step.
