# Findings and limits

Written per `PROMPT.md` section 6. Every number is copied from a run file and tagged with its run id and config hash. Negative results are included. No causal claims beyond what the experiments support.

Status: all three sealed tests are final, each scored once under its lock: v0.1 AMLWorld HI-Small (Section 3) and TabFormer (Section 9), and v0.2 AMLWorld LI-Small (Section 10).

**v0.1 conclusion:** adding TTM summaries of account histories to the bank-style model gave no detectable gain at the pre-registered budget on either sealed test: AMLWorld +0.1 points [-1.6, +1.9] at 0.5%, TabFormer +2.0 points [-0.6, +5.0] at 0.1%. Neither interval excludes zero. Both estimates point the same way and small gains are not ruled out, but v0.1 cannot show them.

**v0.2 conclusion (pre-registered, fresh data):** on AMLWorld LI-Small, adding Chronos-Bolt-tiny summaries gave a modest, statistically significant gain: +3.2 points [+1.1, +5.4], Holm p 0.016, 24.0% against 20.7% of 1,549 newly bad accounts caught on their first day at 0.5%. TTM with recent pooling gave no detectable gain (+0.8 [-1.2, +2.8]). So the v0.1 negative was partly about the model: one different pretrained model added signal, the reworked TTM did not. Caveats in Section 10.

## 1. Question

Does a time series foundation model (TSFM) help a bank-style fraud model catch accounts on the first day they go bad, when analysts can review only a small, fixed share of each day's cases? The TSFM is used as a feature supplier: a frozen model summarizes each account's history, and the summary is added as extra columns to the bank-style model (`docs/methodology.md` M1, M2).

## 2. Setup

- **Data:** AMLWorld HI-Small (synthetic money laundering; 5.08 million transfers, 515,088 accounts, ten usable days after removing an end-of-file artifact) is the main test. IBM TabFormer (synthetic card fraud; 24.4 million transactions, 6,139 cards, 1991 to 2020) is the long-history test. Both measured before use (`docs/data-fit.md`).
- **Unit and budget:** account-day cases; each day the top 0.5% (AMLWorld) or 0.1% (TabFormer) of cases are "reviewed" (M10 to M12).
- **Headline:** share of newly bad accounts caught on their first bad day (accounts already known bad are excluded).
- **Contestants:** A, rank by amount; B, 34 past-only red-flag features plus category and channel into LightGBM; C, TSFM columns only; D, B's inputs plus TSFM columns, with B's exact settings (M20 to M22, M34).
- **TSFM:** IBM TTM r2 (0.8M parameters) on full data; Amazon Chronos-2 (119.5M) only on a 20,000-account sample, because a CPU cannot embed full data in reasonable time (M36). Embeddings are averaged over the history, standardized, and reduced to 16 components by PCA fitted on training days.
- **Honesty machinery:** time split with gaps, past-only features proven by replay, a canary on every run, a label-permutation control, account-level paired bootstrap with Holm correction, and a sealed plan committed before test scoring (M13 to M17, M29 to M31, M37).

## 3. Headline result: AMLWorld, sealed test

Run `20261116T190641Z_amlworld_test_f6ff26be_ac67fd`, config hash `f6ff26be1872`, test run number 1 under the lock committed in `4a21200`. Test period 2022-09-08 to 2022-09-10; 1,798 newly bad accounts.

| Contestant | First-fraud-day catch at 0.5% | Difference vs B [95% interval] | Holm p |
|---|---|---|---|
| A. Lazy baseline | 1.6% | -32.8 [-35.0, -30.7] | 0.006 |
| B. Checklist + LightGBM | 34.4% | reference | |
| C. TTM only | 1.0% | -33.4 [-35.6, -31.2] | 0.006 |
| **D. B + TTM** | **34.5%** | **+0.1 [-1.6, +1.9]** | **0.94** |

**Finding:** adding TTM summaries did not detectably change how many newly bad accounts were caught on their first day at the pre-registered budget. The interval rules out a gain larger than about 2 points.

## 4. What worked

- **The bank-style model is strong, and its probabilities beat always predicting the base rate (Brier score).** On the AMLWorld test it caught 34.4% of newly bad accounts on day one at 0.5%, against 1.6% for ranking by amount.
- **The honesty machinery caught real problems before they became results:** an AMLWorld end-of-file artifact (about 60% laundering in the last days), a dataset whose labels were copied forward (IEEE-CIS), a headline that counted already-known bad accounts, a biased bootstrap, a canary blind spot, and a PCA scaling mistake (`docs/decisions.md`, README "Problems found").
- **The seal protected the conclusion.** Validation showed D +5.7 points ahead after a variant chosen on validation; the sealed test, with 2.7 times as many newly bad accounts, did not confirm it.

## 5. What did not work

- **TTM did not add detectable value at the headline budget on either sealed test** (AMLWorld, Section 3; TabFormer, Section 9).
- **TTM on its own is weak** on every dataset and period (C caught 1.0% on the AMLWorld test and 2.2% on the TabFormer test).
- **The label-permutation control failed on TabFormer validation.** The bank-style model retrained on labels shuffled within each training day should rank no better than random, but it caught 5.9% of newly bad cards against 0.1% for random ranking (+5.8 points [+4.0, +7.8]; run `20261116T193405Z_tabformer_valid_701e5f22_d9a420`). The same control on the TabFormer sealed test was consistent with chance (+0.4 [-0.1, +1.4]). One possible reason is that shuffling within a day keeps each day's fraud rate, so the shuffled model can still learn which kinds of days carry more fraud; a follow-up control that shuffles across all days tests this (`docs/analysis/tabformer-shuffle-scope-control.md`, generated after the sealed runs). Outcome (descriptive): not a leak. Shuffling across all days cut the excess at 0.1% to +1.5 [+0.6, +2.6] and +0.9 [+0.3, +1.6] (two seeds; +5.8 and +4.5 at 0.5%), so day-level bursts were most of it. A model trained on labels shuffled across all days holds no label information, so the remainder comes from the features alone: a case takes its day's highest transaction score, so busier account-days score higher even under noise (newly bad account-days average 4.0 transactions against 2.0; ranking by count alone catches 0.2% at 0.1% and 1.8% at 0.5%), which explains a small part; and a tree fitted to noise still scores rare, unusual-looking transactions highest, which TabFormer fraud tends to be, which plausibly explains the rest (not separately proven). The features passed the past-only proofs and the canary on the same run, and every AMLWorld control passed. The lesson: a constant score is too weak a stand-in for chance on bursty, unusual-looking fraud; a stronger control would compare against many label-free models (methodology M42).
- **The first TTM variant made things worse** on validation (-11.6 points) because of unscaled PCA and a training/evaluation history shift; it is kept on record (run `20261115T031830Z_amlworld_valid_f6ff26be_96919d`).

## 5b. Why the TSFM did not help (descriptive diagnostics)

From `docs/analysis/why-no-gain.md` (generated by `scripts/diagnose_no_gain.py` from the existing runs; not pre-registered), most likely reasons first:

1. **Same-day blindness.** 79% (AMLWorld) and 70% (TabFormer) of the bank-style model's SHAP attribution comes from the transaction itself and the last 24 hours, while the TSFM reads only history before midnight; the first fraud is the day's first transaction for 52% (AMLWorld) and 64% (TabFormer) of newly bad accounts.
2. **Little history on AMLWorld.** 51% of newly bad accounts in the AMLWorld test had fewer than 16 earlier events.
3. **Weak signal alone.** TTM alone reaches about 3.4 times the base-rate PR-AUC, against 100 to 289 times for the bank-style model.
4. **Partial redundancy.** The checklist explains 42% (AMLWorld) and 28% (TabFormer) of the TSFM summary (linear fit, a lower bound).
5. **Reshuffle, not addition.** Caught only with the TSFM versus only without: 140 versus 138 on the AMLWorld test, 10 versus 5 on TabFormer.
6. **Power.** Detectable gain about 2.5 points on AMLWorld (observed +0.1, near zero) and about 4.0 points on TabFormer (observed +2.0, underpowered).

Domain mismatch (pretraining on smooth series, applied to irregular bursts) is a hypothesis, not measured.

## 6. Hints, not findings (descriptive, not pre-registered)

From the AMLWorld sealed test run, D was ahead at the tightest budget (13.6% against 11.0% at 0.1%) and on transaction-level PR-AUC (0.398 against 0.321), and behind at loose budgets (58.8% against 63.1% at 2%; 77.3% against 79.5% at 5%). There was no clear difference in either history group (short: 32.8% against 33.6%; long: 36.2% against 35.2%; not tested). Intervals for every budget, a per-day breakdown, and who-catches-whom counts are in `docs/analysis/amlworld-test-descriptive.md` (generated by `scripts/analyze_test_run.py` after the sealed runs). These suggest TSFM summaries may reshuffle the very top of the queue; testing that would need a new sealed plan with the tight budget as its headline.

On the TabFormer sealed test, D was ahead of B at every budget (Section 9), but only the 0.1% comparison was pre-registered and it was not clearly different from zero. Per-budget intervals are in `docs/analysis/tabformer-test-descriptive.md` (generated after the sealed runs).

**The bank-style model went stale on TabFormer.** B's first-fraud-day catch at 0.5% fell from 90.1% on validation (2016) to 29.9% on the test period (2017 to 2019); at the 0.1% headline it fell from 28.4% to 11.2%. The model was trained on data up to October 2015 and never retrained, so this is consistent with fraud patterns drifting away from what it learned over several years, though the runs do not prove that cause. It supports trying a rolling retrain (Section 8).

## 7. What could not be tested

- **Chronos-2 on full data:** no GPU. On the 20,000-account sample (23 newly bad accounts on validation) its interval was too wide to say anything (+4.3 points, -10.5 to +18.2).
- **Real bank data:** private. All results are on synthetic data from two IBM generators.
- **A training window with established accounts on AMLWorld:** the dataset starts on Sept 1, so training holds almost only new accounts (9.3% with 16 or more earlier events against 62.7% in the test period). No split choice removes this (M41).
- **Graph-based baselines:** out of scope; the bank-style model has money-movement red flags but no transaction-network features.
- **The agent on live Claude:** built and tested with a hand-written cassette; a live run needs an API key (command in `docs/runbook.md`).

## 8. Next steps

1. Done: a pre-registered follow-up on fresh data, Tells v0.2 (result in Section 10) (`docs/v2-plan.md`), testing two variants on untouched AMLWorld LI-Small: `ttm_recent` (recent pooling with more components, for the dilution idea) and `chronos_bolt` (Chronos-Bolt-tiny, a different pretrained model from the family the AMLWorld paper used). MOMENT was planned as the second variant but dropped before any result because it would need about 190 CPU hours (`docs/v2-plan.md`, "Changes before sealing"). It keeps the 0.5% headline on purpose: moving the headline to the budget where a descriptive hint appeared (0.1%) would be choosing the metric after seeing results.
2. Run Chronos-2 on full data with a GPU.
3. Try a rolling retrain: on AMLWorld so training includes established accounts, and on TabFormer, where the bank-style model went stale over the multi-year test period (Section 6).
4. Replicate on real transaction data under a data-sharing agreement.

## 9. TabFormer (long-history test)

### Sealed test (the result that counts)

Run `20261117T045139Z_tabformer_test_701e5f22_2e74f8`, lock `701e5f22` (committed in `4a21200`), test run number 1. Test period 2017-04-01 to 2019-12-31; 243 newly bad cards. Headline budget 0.1%.

| Contestant | First-fraud-day catch at 0.1% | Difference vs B [95% interval] | Holm p |
|---|---|---|---|
| A. Lazy baseline | 1.4% | -9.8 [-13.9, -5.7] | 0.006 |
| B. Checklist + LightGBM | 11.2% | reference | |
| C. TTM only | 2.2% | -9.0 [-12.9, -5.3] | 0.006 |
| **D. B + TTM** | **13.2%** | **+2.0 [-0.6, +5.0]** | **0.16** |

**Finding:** no clear difference between D and B at the pre-registered budget. The canary passed and the label-permutation control was consistent with chance (+0.4 points [-0.1, +1.4]).

Descriptive only, not pre-registered: D was ahead of B at every budget (0.5%: 33.9% against 29.9%; 1%: 47.7% against 43.6%; 2%: 59.9% against 57.2%; 5%: 77.0% against 72.4%) and on transaction-level PR-AUC (0.114 against 0.101). TTM columns made up 6.5% of D's SHAP attribution. Almost every case had a long history (99.5% of training and 99.9% of test account-days with 16 or more earlier events), so the history-length split says nothing here.

### Validation, full data (before the test; not a finding)

Run `20261116T193405Z_tabformer_valid_701e5f22_d9a420`, same config hash. Validation period 2016; 556 newly bad cards.

| Contestant | First-fraud-day catch at 0.1% | Difference vs B [95% interval] | Holm p |
|---|---|---|---|
| A. Lazy baseline | 0.7% | -27.8 [-31.5, -24.2] | 0.006 |
| B. Checklist + LightGBM | 28.4% | reference | |
| C. TTM only | 4.2% | -24.3 [-28.1, -20.4] | 0.006 |
| D. B + TTM | 31.0% | +2.5 [-1.1, +6.0] | 0.16 |

The label-permutation control on this run was not consistent with chance (Section 5). Outcome of the follow-up control: not a leak; shuffling across all days cut the excess to +0.9 to +1.5 points at 0.1%, and the remainder is explained by label-free effects (Section 5, methodology M42).

The prediction stated before testing (any TSFM gain should be larger on AMLWorld than on TabFormer) cannot be judged: neither sealed test showed a clear gain.

## 10. Version 2: AMLWorld LI-Small, sealed test (pre-registered)

Run `20261117T235515Z_amlworld_li_v2_test_cf534a03_0f604f`, config hash `cf534a03d8c1`, test run number 1 under the lock committed in `b8a1927` (both the "started" and "finished" records are in `locks/cf534a03...uses.jsonl`). Test period 2022-09-08 to 2022-09-10; 1,549 newly bad accounts (803 with under 16 earlier events, 746 with 16 or more); 982,049 account-day cases. Plan: `docs/v2-plan.md`. Five comparisons, Holm-corrected.

| Contestant | First-fraud-day catch at 0.5% | Difference vs B [95% interval] | Holm p |
|---|---|---|---|
| A. Lazy baseline | 1.2% | -19.5 [-21.5, -17.3] | 0.010 |
| B. Checklist + LightGBM | 20.7% | reference | |
| C. TTM recent only | 3.7% | -17.0 [-19.0, -14.8] | 0.010 |
| D. B + TTM recent | 21.6% | +0.8 [-1.2, +2.8] | 0.42 |
| C. Chronos-Bolt only | 2.8% | -17.9 [-20.0, -15.7] | 0.010 |
| **D. B + Chronos-Bolt** | **24.0%** | **+3.2 [+1.1, +5.4]** | **0.016** |

**Finding:** D with Chronos-Bolt-tiny meets the pre-registered win rule (Holm p below 0.05, interval above zero, canary and control passing). D with TTM recent pooling does not. Canary passed; label-permutation control +0.0 points [-0.3, +0.4], consistent with chance. Across all four "B + TSFM" sealed tests in the project (v0.1 AMLWorld and TabFormer, v0.2 two variants), a Bonferroni bound on the raw p of 0.008 gives 0.032.

**Descriptive (not pre-registered):**
- By budget, D Chronos-Bolt against B: 6.3% against 6.1% at 0.1%, 37.4% against 36.6% at 1%, 50.7% against 52.9% at 2%, 68.7% against 73.5% at 5%. Ahead at tight budgets, behind at loose ones, as in v0.1 on AMLWorld.
- By history at 0.5%: short 26.2% against 23.5%, long 21.6% against 17.7%.
- Transaction PR-AUC 0.0760 against 0.0733; case PR-AUC 0.0767 against 0.0778.
- The TSFM columns carry 40% (Chronos-Bolt) and 38% (TTM recent) of D's SHAP attribution.
- History mix: 9.3% of training account-days have 16 or more earlier events, 62.8% of test account-days.

**Validation, for context:** on the LI-Small validation day (580 newly bad accounts, scored after sealing), D Chronos-Bolt was -2.1 [-5.9, +1.4] and D TTM recent -3.8 [-7.4, -0.2], neither significant after Holm. The validation day pointed the other way. It is a third the size of the test, and the plan fixed the test as the number that counts before either was scored.

**Limits of this finding:**
- One synthetic dataset. AMLWorld is simulated, so this does not show real-bank performance.
- The effect is modest, about 50 more accounts out of 1,549 at the same budget.
- It is specific to Chronos-Bolt. The reworked TTM did not help, so this is not "TSFMs help".
- Validation disagreed in direction.
- At loose budgets D was behind (descriptive).
- Not yet tested, and the first things an expert would ask (`docs/expert-review.md`, L29 to L31): whether the pretraining causes the gain or simple history statistics would match it; how much training-seed and day-to-day variation widen the interval; and replication beyond the same generator.
- Why Chronos-Bolt helps where TTM did not is not tested. It forecasts with quantiles and reads each channel separately; both are candidate reasons, not shown ones.

