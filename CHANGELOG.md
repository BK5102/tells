# Changelog

All notable changes to Tells. The shared Cleanroom package has its own changelog in `packages/cleanroom/CHANGELOG.md`.

## 0.1.0 (unreleased)

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

#### Sealed test (the result that counts)

AMLWorld HI-Small, test period 2022-09-08 to 2022-09-10, 1,798 newly bad accounts. Run `20261116T190641Z_amlworld_test_f6ff26be_ac67fd`, test run number 1 under lock `f6ff26be`, sealed in commit `4a21200` before scoring.

| Contestant | First-fraud-day catch at 0.5% | Difference vs B [95% interval] | Holm p |
|---|---|---|---|
| A. Lazy baseline | 1.6% | -32.8 [-35.0, -30.7] | 0.006 |
| B. Checklist + LightGBM | 34.4% | reference | |
| C. TTM only | 1.0% | -33.4 [-35.6, -31.2] | 0.006 |
| **D. B + TTM** | **34.5%** | **+0.1 [-1.6, +1.9]** | **0.94** |

Adding TTM summaries of account histories did not detectably change how many newly bad accounts were caught on their first bad day at the pre-registered budget; the interval rules out a gain larger than about 2 points. The canary passed, and the label-permutation control scored like random ranking (+0.1 points, -0.2 to +0.4).

Descriptive only, not pre-registered: D was ahead at the tightest budget (13.6% against 11.0% at 0.1%) and on transaction-level PR-AUC (0.398 against 0.321), and behind at loose budgets (58.8% against 63.1% at 2%, 77.3% against 79.5% at 5%). Results were the same for short-history and long-history accounts.

TabFormer (long-history test): sealed under lock `701e5f22`; the test result is pending.

#### Validation (before sealing; not findings)

These numbers guided development and are kept on record. One setting (standardizing embeddings before PCA) was chosen after seeing the first validation run, which is why only the sealed test can confirm anything.

| | AMLWorld, full data | AMLWorld, 20,000-account sample | TabFormer, 300-card sample |
|---|---|---|---|
| Validation period | 2022-09-06 | 2022-09-06 | 2016 |
| Budget / newly bad accounts | 0.5% / 672 | 0.5% / 23 | 0.1% / 29 |
| B. Checklist + LightGBM | 31.5% | 39.1% | 9.5% |
| D. B + TTM | 37.2%: +5.7 [+2.3, +9.1], p 0.006 | 34.8%: -4.3 [-23.6, +11.8] | 8.7%: -0.8 [-2.2, 0.0], p 0.31 |
| D. B + Chronos-2 | not run (needs a GPU) | 43.5%: +4.3 [-10.5, +18.2] | not run |

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
- **Same-second run folders.** Two runs starting in the same second shared a folder on fast CI machines (`fa5ea99`). Run folders now get a unique suffix and still refuse to overwrite.

### Known limitations

The full register is in `docs/methodology.md` under "Limitations register". The eight that matter most:

1. **Synthetic data only, from two IBM generators** (L1, L2). Nothing here shows how a model performs at a real bank, and generator quirks may drive results.
2. **AMLWorld has ten usable days** (L3). Gaps are one day, validation is a single day, and intervals are wide.
3. **AMLWorld training holds almost only new accounts** (L18). Account-days with 16 or more earlier events are 9.3% of training and 62.7% of the test period, so TSFM features are learned on new accounts and applied to established ones. No split choice removes this.
4. **Chronos-2 ran only on a 20,000-account sample** (L15, L16). Full-data Chronos-2 needs a GPU, so the sealed test covers TTM only.
5. **No graph features in contestant B** (L7). B is weaker than the best published AMLWorld baselines, though its money-movement red flags keep it from being a strawman.
6. **The AMLWorld copy is a Hugging Face mirror** (L4). Its counts match the paper; the checksum comparison with the official Kaggle file is still to do.
7. **The gaps are assumptions** (L6). 90 days for cards and 1 day for laundering stand in for label delay; real delays differ.
8. **The lock does not block code edits** (L13). Git history is what makes any edit between sealing and scoring visible.
