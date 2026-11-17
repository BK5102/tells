# Tells v0.2: pre-registered plan

Date fixed: 2026-11-17. Written before any v0.2 model has been run on any data.

This is a pre-registration: a plan written and committed before the experiment, so nobody (including me) can change the question, the data, or the scoring after seeing a result. Analogy: a referee who writes the rules of the match and seals them in an envelope before kickoff. Anything below marked **V2_TBD** depends on compute or on the data fit check and will be filled in, committed, and sealed by hash with `tells lock` before any model runs on the new data. Nothing marked V2_TBD can be chosen by looking at a result.

Every choice below says what it is, why, what else was considered, and what it causes, with one sentence to say out loud.

## 1. Why a version 2

**What v0.1 found.** Tells v0.1 sealed two test plans and scored each once. Adding TTM summaries (a small time series foundation model from IBM) to the bank-style LightGBM model gave no detectable gain in newly bad accounts caught on their first bad day at the pre-registered budget:

| Sealed test | Budget | B. Bank-style | D. B + TTM | D minus B [95% interval] | Holm p |
|---|---|---|---|---|---|
| AMLWorld HI-Small, Sept 8 to 10 | 0.5% | 34.4% | 34.5% | +0.1 [-1.6, +1.9] | 0.94 |
| TabFormer, April 2017 to 2019 | 0.1% | 11.2% | 13.2% | +2.0 [-0.6, +5.0] | 0.16 |

Runs: `20261116T190641Z_amlworld_test_f6ff26be_ac67fd` and `20261117T045139Z_tabformer_test_701e5f22_2e74f8`. Details: `docs/findings.md`.

**Why not stop there.** v0.1 answered its question honestly, and that answer stands. But looking at how the TSFM columns were built (after the tests, as diagnosis, not as a result) points to specific, testable reasons the signal could have been diluted before it ever reached the tree model:

1. The embedding was an average over every patch (a patch is a fixed-size chunk of the history, like one page of a bank statement). Fraud is a change in behavior. Averaging a whole history blurs a recent change into the old pattern.
2. TTM reads a fixed 512-step window and left-pads short histories (M35). AMLWorld accounts have a median of about 10 events in that window, so most patches are padding. Averaging over them waters down the few patches that hold real activity.
3. Keeping only 16 PCA components may drop rare patterns, and fraud is a rare pattern.
4. The descriptive hints on the sealed AMLWorld test (D ahead at the tightest 0.1% budget and on transaction-level PR-AUC) are not findings, but they are consistent with a weak signal that a better summary might strengthen.

**Why not test these on the v0.1 test sets.** Both v0.1 test periods have been seen. Trying new variants on them until one wins would be p-hacking: with enough tries, one variant beats B by luck alone, and the result would look like a finding without being one. Analogy: re-sitting the same exam with the answer key in your head.

**Why a new pre-registration is the honest route.** A follow-up question deserves a fresh exam: a new plan, fixed by reasoning before any result, scored once on data no model has touched. That turns "maybe it would have worked if..." into a test that can come out either way.

**Effect on the rest of the project.** The shared Cleanroom package does not change, so the sibling projects (Reserve and Culprit) are not affected and no new `cleanroom` tag is needed. The new TSFM adapter follows the existing adapter interface in `src/tells/models/tsfm/`, so it is reusable. Keeping v0.1 sealed and closed protects the credibility of all three projects, because they rely on the same promise: a sealed test is scored once.

- **What else I considered:** stopping at v0.1 (honest, but leaves specific, cheap-to-test explanations unexamined); re-running new variants on the v0.1 test periods (fishing); calling the v0.1 hints a result (they were not pre-registered).
- **How to say it:** "v0.1 gave an honest negative. Diagnosing it suggested two specific reasons the signal might have been watered down, and the only honest way to test them was a new sealed plan on fresh data, not more tries on a test I had already used."

## 2. The two ideas, and why each

Exactly two variants, fixed now by reasoning, not by results. Each gets its own C (TSFM columns only) and D (bank-style columns plus TSFM columns).

### Variant 1: `ttm_recent`

- **What:** the same TTM r2 model and the same input series as v0.1 (the account's last 256 events before midnight, as log amount, log time since the previous event, and direction). Two changes only: the embedding is taken from the **most recent patch** instead of the average over all patches, and **64** standardized PCA components are kept instead of 16 (standardize, then PCA fitted on training days only, as in v0.1).
- **Why the most recent patch:** fraud is a change, and the most recent patch is where a change shows first. Because the window is left-padded, the most recent patch is also the one that always holds real events, even for a short history. This removes the padding dilution and the blurring in one step.
- **Why 64 components:** 16 components may throw away rare directions in the embedding; 64 keeps more of them while staying small enough for LightGBM on a CPU. 64 is fixed now and not tuned.
- **What else I considered:** max pooling (picks extremes, but also noise from padding); averaging only over non-padded patches (fixes padding but still blurs the recent change); the last few patches (adds a tuning choice, more chances to fish); keeping all 192 x 3 dimensions (slow, and more columns than the training days can support well). Each would be a reasonable variant, but adding them would multiply comparisons and weaken the correction.
- **Cause and effect:** if v0.1's null came from dilution, this variant should move D above B. If it does not, dilution was not the main reason, which is also worth knowing.
- **How to say it:** "Fraud is a change in behavior, so I stopped averaging the whole history, most of which was padding, and used the summary of the most recent stretch instead."

### Variant 2: `moment`

- **What:** MOMENT-1 (Goswami et al. 2024, "MOMENT: A Family of Open Time-series Foundation Models", arXiv 2402.03885), from Carnegie Mellon's Auton Lab, MIT license. It is a TSFM designed to produce embeddings and support classification and anomaly detection, not only forecasting. It runs in embedding mode on the same input series and 512-step window, with padding marked by its input mask. The downstream setup matches variant 1: 64 standardized PCA components fitted on training days only, into the same LightGBM with B's exact settings.
- **Model size and revision:** V2_TBD (chosen by measured CPU cost before sealing, pinned to an exact Hugging Face commit as in v0.1).
- **Why:** TTM and Chronos-2 were built mainly to forecast the next values. A model trained to represent whole series may summarize behavior better, and summarizing behavior is the job here. This tests "was it the wrong kind of model?" separately from "was it the wrong summary?".
- **No surprise score:** MOMENT gives no forecast in embedding mode, so this variant has no surprise columns. That makes its C and D slightly different from TTM's in what they contain; the difference is stated, not hidden.
- **What else I considered:** Chronos-2 on full data (no GPU, see Section 7); other embedding-oriented models (each would add a comparison; MOMENT is open, documented for embeddings, and from a different lab than TTM and Chronos-2).
- **Cause and effect:** if MOMENT helps and `ttm_recent` does not, the model's training goal matters more than the pooling. If neither helps, the evidence against frozen TSFM summaries for this task gets stronger.
- **How to say it:** "The models I had used were built to forecast, so I added one built to summarize series, to test whether the kind of model was the problem."

## 3. What is fixed now

These do not change after this date, and the v0.2 config holds them before it is sealed:

- The two variants above, their pooling, and 64 components. No other variant will be run on the new data.
- Contestants A, B, C x 2, D x 2 (Section 5), with B's LightGBM settings unchanged from v0.1 and no early stopping.
- The headline metric, budget, bootstrap, and correction (Section 5).
- The data, split, and gaps (Section 4), subject only to the data fit check.
- The success rule (Section 6) and the reporting rule (Section 8).
- The canary and the label-permutation control run on every v0.2 run, as in v0.1. A run whose canary fails does not count.

**Where development happens:** only on AMLWorld HI-Small's validation day (2022-09-06), which v0.1 already used. It is used to check that the code works and that the variants run end to end, not to pick between them. No setting is tuned on LI-Small. HI-Small's test period (Sept 8 to 10) is not used again.

- **Why develop on HI-Small:** the code must be checked somewhere, and a day that is already spent cannot be spent again. Keeping all checks there means LI-Small arrives untouched at sealing time.
- **How to say it:** "I debugged on data I had already used, so the new data stayed untouched until the plan was sealed."

## 4. Data and split

- **Dataset:** AMLWorld LI-Small. Same IBM generator as HI-Small, with a lower laundering rate ("LI" is low illicit, "HI" high illicit). No Tells model has been run on it.
- **Why this dataset:** it is the only fresh data that asks the same question under the same rules. No untouched TabFormer period exists (v0.1 used 2016 for validation and April 2017 to 2019 for testing, and 2020 has no fraud). A lower laundering rate is also closer to real banks, where laundering is rarer.
- **Split:** the same calendar split as v0.1: train Sept 1 to 4, validate Sept 6, test Sept 8 to 10, with one-day gaps, through `cleanroom.split`. This holds only if the data fit check confirms LI-Small has enough newly bad accounts per period and no end-of-file artifact inside those dates.
- **Truncation date:** V2_TBD (from the data fit check, as Sept 11 onward was dropped from HI-Small because of an artifact). If the fit check forces a different split, it is changed before sealing, written here, and the reason recorded; it is never changed after.
- **Order of events:** fit check on LI-Small, then fill in the V2_TBD items, then `tells lock` on the v0.2 config, then commit the lock, then score LI-Small validation, then score LI-Small test once. Validation is scored after sealing too, so it cannot influence any setting.
- **What else I considered:** HI-Medium or larger AMLWorld files (more compute than a CPU can embed with MOMENT in reasonable time); holding out a slice of HI-Small (every usable HI-Small day is already in v0.1's train, validation, or test).
- **Limitation:** same generator as HI-Small, so a result says "on a second synthetic laundering world", not "at a real bank". The training-history shift found in v0.1 (M41: training days hold mostly brand-new accounts) is expected here too, because LI-Small also starts on Sept 1; every run reports the history mix.
- **How to say it:** "The new test is a sister dataset from the same generator that no model of mine had ever touched, split on the same calendar, and sealed before I looked."

## 5. Contestants and headline

| | Contestant | What it is |
|---|---|---|
| A | Lazy baseline | Rank by amount |
| B | Bank-style (reference) | 34 past-only red-flag features plus category and channel into LightGBM |
| C_ttm_recent | TSFM only | `ttm_recent` columns (64 components plus TTM's surprise scores and has-history flag) |
| D_ttm_recent | B + TSFM | B's columns plus C_ttm_recent's columns, B's exact settings |
| C_moment | TSFM only | `moment` columns (64 components plus has-history flag; no surprise score) |
| D_moment | B + TSFM | B's columns plus C_moment's columns, B's exact settings |

- **Headline (unchanged from v0.1 AMLWorld):** first-fraud-day catch rate at a 0.5% daily review budget, D vs B, paired bootstrap over whole accounts, 1,000 resamples.
- **Correction:** Holm across all five comparisons with B (A, both Cs, both Ds). Variants tried across the project are counted and reported: two in v0.1 (unstandardized and standardized PCA), two in v0.2.
- **Why keep 0.5% instead of moving to 0.1%:** v0.1's findings listed "a new sealed plan with 0.1% as the headline" as a next step, because D was ahead at 0.1% on the sealed AMLWorld test. Moving the headline to the budget where a hint appeared would be choosing the metric after seeing the data. Keeping 0.5% makes v0.2 a clean follow-up to v0.1's exact question. Every other budget, including 0.1%, is reported next to it as descriptive.
- **How to say it:** "Same contestants, same headline, same budget as before, so the only thing that changed is how the account history is summarized."

## 6. What counts as success

- **A win:** at least one D variant beats B on the LI-Small sealed test headline with a Holm-adjusted p below 0.05 and a 95% bootstrap interval above zero (the same rule the Critic applies, M40). The canary and label-permutation control must pass on that run.
- **Not a win:** a D ahead of B with an interval that includes zero; a D ahead only at budgets other than 0.5%; a D ahead on LI-Small validation but not on test; any gain on HI-Small validation. These are reported as descriptive.
- **A loss:** a D below B with Holm p below 0.05 is reported as harm.
- **A second negative result is a legitimate outcome** and is reported the same way as a win. It would say that two specific, reasoned fixes did not rescue frozen TSFM summaries on this task.
- **How to say it:** "I wrote down before the run what would count as a win. A second honest negative counts as an answer, not a failure."

## 7. Not in v0.2, and why

| Left out | Why |
|---|---|
| Chronos-2 on full data | No GPU. Measured at about 66 ms per account-day on AMLWorld on CPU (M36), plus its forecast; full data would take days. Still a next step with a GPU |
| A new TabFormer test | No untouched TabFormer period exists: 2016 and April 2017 to 2019 are spent, and 2020 has no fraud |
| Graph features | They answer a different question (does the transaction network help?), not whether account histories add to a bank-style model |
| A supervised sequence model (trained end to end on fraud labels) | A different approach from frozen TSFM summaries, more expensive on CPU, and more choices to tune. Kept for later, under its own plan |
| More pooling or size variants | Each extra variant adds a comparison and weakens the correction; two reasoned variants is a sharper test than many guessed ones |
| A 0.1% headline | Chosen from a v0.1 hint; see Section 5 |

## 8. How the result will be reported, either way

- The LI-Small test result goes into `docs/findings.md`, the README, and `docs/explain-the-project.md`, in the same table format as v0.1, whatever it shows.
- Numbers come from result files only (CLAUDE.md rule 4), with run id, config hash, lock hash, and test run number.
- Every budget, both history groups, transaction-level PR-AUC, SHAP share of TSFM columns, the canary, and the label-permutation control are reported next to the headline, labeled descriptive.
- v0.1's sealed results are not changed, re-scored, or re-described in light of v0.2.
- If the test must be re-scored for any reason (a crash, a bug), the run count goes up and is reported; a re-score after seeing a result needs a new lock and is reported as such.

## Placeholders to fill before sealing

| Item | Status |
|---|---|
| MOMENT size (small, base, or large) | V2_TBD |
| MOMENT Hugging Face revision (exact commit) | V2_TBD |
| LI-Small truncation date and fit check result | V2_TBD |
| v0.2 config path | V2_TBD |
| Lock hash and commit | recorded at sealing |
