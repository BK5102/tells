# Why the TSFM features gave no detectable gain (Tells v0.1)

Descriptive and explanatory, not pre-registered, and not a search for a win: no contestant was retrained and the test period was not scored again. Sources: the four key runs' `results.json` and case files (sealed test case files used only for what they already contain), the validation periods of the prepared data, and the TTM embedding caches. Every number is computed by `scripts/diagnose_no_gain.py`. Runs: AMLWorld validation `20261115T041547Z_amlworld_valid_f6ff26be_5e9b71`, AMLWorld sealed test `20261116T190641Z_amlworld_test_f6ff26be_ac67fd`, TabFormer validation `20261116T193405Z_tabformer_valid_701e5f22_d9a420`, TabFormer sealed test `20261117T045139Z_tabformer_test_701e5f22_2e74f8`.

Terms used below. **B** is the bank-style model (checklist red flags into LightGBM). **D** is B plus TTM columns (16 PCA components of the TTM summary of each account's history before midnight, two surprise scores, a has-history flag). **C** is the TTM columns alone. A **newly bad account** is one whose first fraud day falls in the evaluated period; the headline is the share of them caught on that first day within the daily review budget.

## 1. Statistical power: how small a gain could each test see?

The 95% interval half-width is roughly the smallest difference that, if observed, would have been clearly separated from zero. A true gain needs to be about 1.4 times that (2.80 / 1.96 standard errors) to be detected 80% of the time. Converting points to accounts shows how few extra catches a gain would mean.

| Run | Budget | Newly bad | B | D | D minus B [95% interval] | Half-width | Gain needed for 80% power | That gain in accounts |
|---|---|---|---|---|---|---|---|---|
| AMLWorld sealed test | 0.5% | 1,798 | 34.4% | 34.5% | +0.1 [-1.6, +1.9] | 1.7 pts | 2.5 pts | about 45 of 1,798 |
| TabFormer sealed test | 0.1% | 243 | 11.2% | 13.2% | +2.0 [-0.6, +5.0] | 2.8 pts | 4.0 pts | about 10 of 243 |
| AMLWorld validation | 0.5% | 672 | 31.5% | 37.2% | +5.7 [+2.3, +9.1] | 3.4 pts | 4.8 pts | about 33 of 672 |
| TabFormer validation | 0.1% | 556 | 28.4% | 31.0% | +2.5 [-1.1, +6.0] | 3.6 pts | 5.1 pts | about 28 of 556 |

**What it means.** AMLWorld could see a gain of about 1.7 points (31 more of its 1,798 newly bad accounts); the observed difference was +0.1 points, about 2 accounts, so on AMLWorld the TSFM effect at the headline budget is genuinely close to zero, not hidden by noise. TabFormer had only 243 newly bad cards, so its half-width is 2.8 points and a real gain would need to be about 4.0 points (about 10 extra cards) to be detected reliably. Its observed +2.0 points (about 5 cards) is below that line: on TabFormer, "no detectable gain" is partly a power problem. Analogy: a bathroom scale that reads in 3-kilogram steps cannot tell you whether you gained one kilogram.

## 2. Reshuffle versus add: does D catch new accounts or different ones?

Same rule as `scripts/analyze_test_run.py`: a newly bad account is caught when at least half of its review slot is used (`cleanroom.metrics.budget_selection`, per day, ties share slots). Queue overlap is the share of B's reviewed cases that D also reviews.

| Run | Budget | Both | Only B | Only D | Neither | Score correlation (Pearson / Spearman) | Queue overlap |
|---|---|---|---|---|---|---|---|
| AMLWorld sealed test | 0.5% | 480 | 138 | 140 | 1,040 | 0.773 / 0.789 | 53.4% |
| TabFormer sealed test | 0.1% | 19 | 5 | 10 | 209 | 0.703 / 0.954 | 46.1% |
| AMLWorld validation | 0.5% | 159 | 53 | 91 | 369 | 0.689 / 0.716 | 50.9% |
| TabFormer validation | 0.1% | 86 | 53 | 63 | 354 | 0.960 / 0.955 | 54.4% |

**What it means.** On the AMLWorld test, D gained 140 accounts B missed and lost 138 that B caught, so the net change is close to zero while 278 accounts changed hands. On TabFormer D gained 10 and lost 5: more gained than lost, but still a swap rather than a pure addition. The reviewed queues themselves differ a lot: only 53% (AMLWorld) and 46% (TabFormer) of the cases B reviews are also reviewed by D, even though the two models' scores correlate strongly overall. Adding TTM columns does not stack new catches on top of B's; it swaps some catches for others, because the extra columns change which cases fill a fixed number of review slots. Analogy: a new scout on a team with a fixed roster size; every player the scout adds pushes someone else off the roster, and the team gets better only if the newcomers are better on average than the ones they replace.

## 3. Redundancy: how much of the TTM summary does the checklist already know?

For a random sample of 40,000 validation account-days per dataset (seed 0; only those with some history kept), each TTM PCA component was regressed on the account-day's checklist features. PCA was refitted on training days with the harness's own `fit_pca`; the share of variance it keeps matches the run's recorded value, which confirms these are the same components the contestants used. Checklist features are averaged over the day's transactions (main result) or taken from the day's first transaction (whose history is closest to the TSFM's midnight cut). Each of the 34 checklist columns enters as a signed log, its square, and a missing flag where needed; the fit is ordinary least squares, so R squared is a lower bound: a flexible model could explain at least this much. Category and channel are not included.

**AMLWorld** (39,894 account-days, 58 predictors; PCA keeps 96.8% of the variance, run recorded 96.8%):

| Component | Share of the 16 components' variance | Adjusted R squared (day mean) | Adjusted R squared (first transaction) | R squared from history length alone |
|---|---|---|---|---|
| PC0 | 20.0% | 0.81 | 0.85 | 0.30 |
| PC1 | 40.6% | 0.22 | 0.23 | 0.02 |
| PC2 | 13.2% | 0.49 | 0.40 | 0.01 |
| PC3 | 2.9% | 0.52 | 0.50 | 0.22 |
| PC4 | 3.0% | 0.34 | 0.21 | 0.02 |
| All 16, variance-weighted | 100% | 0.42 | 0.41 | 0.11 |

**TabFormer** (39,994 account-days, 68 predictors; PCA keeps 85.2% of the variance, run recorded 85.2%):

| Component | Share of the 16 components' variance | Adjusted R squared (day mean) | Adjusted R squared (first transaction) | R squared from history length alone |
|---|---|---|---|---|
| PC0 | 26.5% | 0.38 | 0.38 | 0.23 |
| PC1 | 25.9% | 0.47 | 0.46 | 0.35 |
| PC2 | 20.0% | 0.29 | 0.27 | 0.06 |
| PC3 | 2.3% | 0.03 | 0.03 | 0.01 |
| PC4 | 2.0% | 0.01 | 0.01 | 0.00 |
| All 16, variance-weighted | 100% | 0.28 | 0.28 | 0.16 |

**What it means.** A high R squared means that component is mostly a re-description of facts the checklist already has (how many earlier events, typical amounts, gaps between transactions). Weighted by variance, the checklist explains 42% of the TTM summary on AMLWorld and 28% on TabFormer with a linear fit, so the overlap is moderate on AMLWorld and modest on TabFormer, not total. Some single components are close copies (AMLWorld PC0: 0.81), and history length alone explains 11% (AMLWorld) and 16% (TabFormer) of the summary. The rest is information the checklist does not hold in linear form (a flexible model might find more of it), but section 5 shows that the TTM summary as a whole carries little fraud signal, and section 4 shows that in D the TTM columns mainly took attribution away from the checklist's history features. On AMLWorld the component with the most variance on these validation days is PC1, not PC0, although PCA orders components by their variance on training days: another sign of the training versus evaluation history shift (methodology M41). Analogy: a second witness who partly repeats the first and otherwise talks about things unrelated to the crime adds little to the case.

## 4. Same-day blindness: the TSFM cannot see the day it is scoring

The TTM summary is cut at midnight (methodology M32), so it knows nothing about the first bad day itself. Newly bad accounts and their histories, from the case files (`history_len` is the number of events before the account's first transaction that day):

| Run | Newly bad | With fewer than 16 prior events | Median prior events | All cases with fewer than 16 | Median transactions on first bad day | First bad day with one transaction |
|---|---|---|---|---|---|---|
| AMLWorld validation | 672 | 67.7% | 9 | 51.6% | 3 | 24.9% |
| AMLWorld sealed test | 1,798 | 51.2% | 15 | 37.3% | 3 | 25.8% |
| TabFormer validation | 556 | 0.7% | 2,323 | 0.1% | 4 | 5.8% |
| TabFormer sealed test | 243 | 0.0% | 2,243 | 0.1% | 2 | 23.5% |

Where the first fraud transaction sits within the first bad day (validation periods only, transaction labels; "before" counts transactions with a strictly earlier timestamp, which is what the checklist can see):

| Dataset (validation) | Newly bad | Median transactions that day | Median transactions before the first fraud one | First fraud is the day's first transaction | At most 2 before | First fraud shares its timestamp with another transaction |
|---|---|---|---|---|---|---|
| AMLWorld | 672 | 3 | 0 | 52.1% | 73.8% | 19.5% |
| TabFormer | 556 | 4 | 0 | 64.4% | 93.9% | 1.3% |

Where B's and D's attribution comes from (mean absolute SHAP shares on up to 20,000 evaluated transactions, from `results.json`). "This transaction" is the transaction's own facts (amount, hour, weekday, direction, category, channel, flags); "short window" is the last 24 hours or this transaction against the past (seconds since previous, counts and sums over 1 and 24 hours, new merchant or category, new counterparties in 24 hours, money in and out over 24 hours); "long history" is 7- and 30-day and lifetime summaries.

| Run | B: this transaction | B: short window | B: long history | D: this transaction | D: short window | D: long history | D: TTM columns |
|---|---|---|---|---|---|---|---|
| AMLWorld validation | 49.4% | 30.1% | 20.5% | 50.0% | 21.9% | 11.5% | 16.5% |
| AMLWorld sealed test | 48.8% | 29.9% | 21.3% | 48.9% | 22.2% | 12.2% | 16.8% |
| TabFormer validation | 57.6% | 12.3% | 30.1% | 53.3% | 12.1% | 28.0% | 6.6% |
| TabFormer sealed test | 57.6% | 12.5% | 30.0% | 53.3% | 12.3% | 27.8% | 6.5% |

**What it means.** On AMLWorld, B draws 79% of its attribution from the transaction itself and the last 24 hours: the part a midnight-cut history summary cannot see (today) or sees only blurred into a long average (the hours before midnight). In AMLWorld validation, 52% of newly bad accounts' first fraud transaction is the first transaction of the day (TabFormer: 64%), so the decisive evidence is usually the fraud transaction's own facts, which B already reads. And 51% of AMLWorld test newly bad accounts had fewer than 16 earlier events (16 events is one Chronos-2 patch; TTM reads a 512-step window in 64-step patches, so for these accounts its input is almost all padding, M35 and M45), so for many of them the summary has little to read. TabFormer is the opposite case: long histories (none of the test's newly bad cards had fewer than 16), and that is where D's point estimate was a little better. Analogy: a background check run last night cannot flag what happens at the counter this morning; the cashier watching the counter can.

The attribution table also shows where D's TTM share (16.8% on the AMLWorld test) came from: the history features' share fell from 51.2% in B to 34.3% in D, while the transaction's own facts stayed at 48.8% and 48.9%. On the TabFormer test the TTM share was 6.5% and the history share fell from 42.4% to 40.2%. The TTM columns took over work the checklist's history features were already doing, which fits the redundancy in section 3.

## 5. How much signal is in the TSFM alone?

Contestant C uses only the TSFM columns. Random ranking catches exactly the budget share. Every run with a C contestant is listed, including development runs on samples and the first, unstandardized TTM variant.

| Run | Part | Contestant | Budget | Newly bad | A (amount) | Random | C | B | C transaction PR-AUC | B transaction PR-AUC | Fraud base rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `20261115T031830Z_amlworld_valid_f6ff26be_96919d` | valid | C_tsfm_only_ttm | 0.5% | 672 | 2.1% | 0.5% | 4.8% | 31.5% | 0.008 | 0.409 | 0.0011 |
| `20261115T041547Z_amlworld_valid_f6ff26be_5e9b71` | valid | C_tsfm_only_ttm | 0.5% | 672 | 2.1% | 0.5% | 3.0% | 31.5% | 0.013 | 0.409 | 0.0011 |
| `20261115T042711Z_small_valid_0f1dcdc6_df305f` | valid | C_tsfm_only_ttm | 0.1% | 29 | 1.3% | 0.1% | 0.0% | 9.5% | 0.006 | 0.776 | 0.0020 |
| `20261115T212743Z_amlworld_small_valid_feecada6_92f7a2` | valid | C_tsfm_only_ttm | 0.5% | 23 | 4.3% | 0.5% | 4.3% | 39.1% | 0.009 | 0.247 | 0.0011 |
| `20261115T212743Z_amlworld_small_valid_feecada6_92f7a2` | valid | C_tsfm_only_chronos2 | 0.5% | 23 | 4.3% | 0.5% | 4.3% | 39.1% | 0.038 | 0.247 | 0.0011 |
| `20261116T190641Z_amlworld_test_f6ff26be_ac67fd` | test | C_tsfm_only_ttm | 0.5% | 1,798 | 1.6% | 0.5% | 1.0% | 34.4% | 0.004 | 0.321 | 0.0011 |
| `20261116T193405Z_tabformer_valid_701e5f22_d9a420` | valid | C_tsfm_only_ttm | 0.1% | 556 | 0.7% | 0.1% | 4.2% | 28.4% | 0.020 | 0.842 | 0.0021 |
| `20261117T045139Z_tabformer_test_701e5f22_2e74f8` | test | C_tsfm_only_ttm | 0.1% | 243 | 1.4% | 0.1% | 2.2% | 11.2% | 0.004 | 0.101 | 0.0010 |

**What it means.** On the sealed tests C caught 1.0% on AMLWorld (random 0.5%, ranking by amount 1.6%, B 34.4%) and 2.2% on TabFormer (random 0.1%, ranking by amount 1.4%, B 11.2%). On the AMLWorld test the TSFM alone is below simply ranking by amount. Its transaction PR-AUC is 3.4 (AMLWorld) and 3.5 (TabFormer) times the fraud base rate, against 289 and 100 times for B. So the summary carries some signal, but far less than the checklist, and a feature that weak can only help B if its signal is also different from what B already has. Analogy: a smoke detector that rarely beeps for real fires adds little to a building that already has sprinklers in every room.

## 6. Domain mismatch (cited hypothesis, not measured)

`docs/tsfm-notes.md` records Chronos-2's documented pretraining data as energy, traffic, weather, web, cloud traces, M4, and synthetic series, with no financial or transaction data listed; TTM r2's pretraining data was not checked in a paper. Those are mostly regularly sampled series with daily or weekly rhythms (our characterization, not a claim from the notes). Account histories are irregular: events arrive in bursts, with long silences between them, and the project feeds them as one step per event with the time gap as a separate channel (methodology M32, M35). The hypothesis is that a model pretrained on smooth, evenly spaced series summarizes such bursty event sequences poorly, so its summary falls back to coarse facts such as how long and how large the history is (consistent with section 3, but not proven by it). This was not measured here.

## 7. The label-permutation control: how easily is unusual fraud ranked without labels?

Built-in control (B retrained on labels shuffled within each training day), from each run's `results.json`, at the headline budget:

| Run | Noise-trained model | Random | Difference [95% interval] | Consistent with chance |
|---|---|---|---|---|
| AMLWorld validation | 0.3% | 0.5% | -0.2 [-0.5, +0.2] | yes |
| AMLWorld sealed test | 0.6% | 0.5% | +0.1 [-0.2, +0.4] | yes |
| TabFormer validation | 5.9% | 0.1% | +5.8 [+4.0, +7.8] | no |
| TabFormer sealed test | 0.5% | 0.1% | +0.4 [-0.1, +1.4] | yes |

Shuffle-scope follow-up on TabFormer validation at the 0.1% headline budget (parsed from `docs/analysis/tabformer-shuffle-scope-control.md`):

| Shuffle | Noise-trained model | Random | Difference [95% interval] |
|---|---|---|---|
| within_day | 5.9% | 0.1% | +5.8 [+4.0, +7.7] |
| across_all | 1.6% | 0.1% | +1.5 [+0.6, +2.6] |
| across_all_seed1 | 1.0% | 0.1% | +0.9 [+0.3, +1.6] |

**What it means.** Even with labels shuffled across all training days, so no real link between behavior and fraud is left, LightGBM catches 1.0% to 1.6% of newly bad TabFormer cards at 0.1%, against 0.1% for random ranking. One explanation consistent with this (not proven here): a tree fitted to noise still gives extreme scores to rare combinations of feature values, so it behaves like an anomaly detector, and fraud on TabFormer often looks unusual. For comparison, C (TTM alone, trained on real labels) caught 4.2% on the same period, between the across-all-days noise models and the within-day one (5.9%): the TSFM's standalone signal is of the same order as what a tree finds with no real labels at all. On AMLWorld the noise-trained model was consistent with chance, so this reading applies to TabFormer only. Two consequences: unusual-looking fraud is already near the top of B's queue without any help, and a new feature can only add catches among the fraud that does not look unusual, which is a smaller and harder pool. Analogy: if a guard dog barks at every stranger, a second dog that also barks at strangers adds nothing; the help you need is with intruders who look like residents.

## Most likely reasons, ranked (interpretation of the numbers above)

The ranking is a judgment about how directly each number bears on the headline; the reasons overlap and none was tested as a cause.

1. **The evidence that decides a first bad day is today's, which the TSFM cannot see.** On the AMLWorld test B draws 79% of its attribution from the transaction itself and the last 24 hours (TabFormer: 70%), and in validation the first fraud transaction was the day's first transaction for 52% (AMLWorld) and 64% (TabFormer) of newly bad accounts (section 4).
2. **The TTM summary alone carries little fraud signal.** C caught 1.0% on the AMLWorld test (amount ranking 1.6%, B 34.4%) and 2.2% on the TabFormer test (B 11.2%) (section 5).
3. **In a fixed queue, D's new catches were cancelled by lost ones.** At the AMLWorld headline D gained 140 and lost 138 newly bad accounts (TabFormer: gained 10, lost 5), while only 53% and 46% of B's reviewed cases stayed in D's queue (section 2). This is how the null shows up rather than a root cause, but it explains why a changed queue gave an unchanged headline.
4. **Part of the TTM summary repeats the checklist, and D used it as a substitute.** The checklist explains 42% (AMLWorld) and 28% (TabFormer) of the components' variance linearly (section 3), and in D the history features' attribution fell from 51.2% to 34.3% on the AMLWorld test as the TTM columns took 16.8% (section 4).
5. **On AMLWorld, many newly bad accounts have short histories.** 51% had fewer than 16 earlier events (median 15), so a history summary has little to read (section 4); training-period accounts are shorter still (methodology M41).
6. **On TabFormer, the test was too small to see a modest gain.** 243 newly bad cards give a half-width of 2.8 points; detecting a gain reliably needed about 4.0 points, and the observed difference was +2.0 (section 1). This does not apply to AMLWorld, where the half-width was 1.7 points and the estimate +0.1.
7. **Unusual-looking fraud is already easy to rank (TabFormer).** Models trained on labels shuffled across all days still caught 1.0% to 1.6% at the 0.1% budget, and TTM alone caught 4.2% (section 7), which leaves less room for any new feature.
8. **Domain mismatch between pretraining series and transaction histories** (section 6). Plausible and consistent with the above, but not measured.

## What would have to change for a TSFM to help (proposals, not evidence)

These follow from the reasons above; none has been tested in this project.

- **See the day being scored.** Embed the sequence up to each transaction (or each hour) instead of up to midnight, so the summary can react to a same-day burst. This costs one embedding per transaction instead of per account-day.
- **Learn from fraud labels.** Fine-tune the TSFM, or train a small head on its patch states, so the summary is shaped toward fraud rather than toward forecasting energy or traffic.
- **Keep the order and the recent past.** Average pooling over the whole history blurs a recent change; recent-patch pooling with more components is v0.2 idea 1 (M45).
- **Larger models on full data,** which needs a GPU (Chronos-2 ran only on a sample, M36).
- **Data with longer and more realistic histories,** ideally real transaction data under a data-sharing agreement, where an account's past is long enough to summarize and fraud is less obviously unusual.
- **Enough newly bad accounts to see a few points of gain,** or a pre-registered tighter budget where section 1 of `docs/analysis/amlworld-test-descriptive.md` hinted at a difference.

