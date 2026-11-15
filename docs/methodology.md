# Methodology: every choice, why it was made, and what it costs

This file is the reasoning behind Tells. Each decision has the same four parts: **what** was chosen, **why**, **what else was considered**, and the **limitation** it brings. Decision ids (M1, M2, ...) are stable so other docs can point to them. `docs/decisions.md` is the dated diary of when things were built; this file is the argument for why they were built that way.

Rule for contributors (also in `CLAUDE.md`): any new method, variable, threshold, or default gets an entry here before it is merged.

---

## 1. The question

**M1. The question is "does a TSFM add value on top of what banks already use, at a fixed review capacity?", not "can a TSFM detect fraud?"**
- Why: banks already run tree models on hand-built features. A new model only matters if it adds something to that, and only inside the number of cases analysts can actually review.
- Considered: comparing a TSFM to nothing, or to a weak baseline only. Rejected because beating a strawman proves nothing.
- Limitation: the answer is about added value under these datasets and this baseline, not about TSFMs in general.

**M2. The TSFM is tested as a feature supplier, not as a stand-alone detector.**
- Why: published work that reports gains uses frozen TSFM embeddings as extra inputs ([Savvidi et al. 2026](https://arxiv.org/html/2610.29667v1)). It is also how a bank would adopt it: add columns, keep the model it already validated.
- Considered: fine-tuning the TSFM end to end. Deferred: costly on CPU, and harder to explain and audit.
- Limitation: a fine-tuned TSFM might do better than frozen features; that is not tested.

## 2. Data

**M3. Only public synthetic data.**
- Why: real transaction data is private and personal. Synthetic data can be shared, rerun, and audited by anyone.
- Limitation: synthetic data is cleaner than reality. No result here says how a model performs at a real bank.

**M4. AMLWorld HI-Small is the main test; TabFormer is the secondary, long-history test.**
- Why AMLWorld is the main test: (1) contestant B, the bank-style model, is far from perfect on it (it caught 37% of first-day laundering at a 0.5% budget on the development sample, against about 96% of card fraud cases on TabFormer), so a TSFM gain can actually show; (2) it is the benchmark the published TSFM paper used, so results can be set next to theirs; (3) its accounts have short histories (median 9 events before the first laundering), which tests whether a pretrained model helps where a bank has little data on a customer.
- Why TabFormer stays: it is the opposite case, cards with years of history (median 1,493 transactions before the first fraud). If the TSFM helps on one and not the other, history length is the first explanation to test.
- Considered: IEEE-CIS, PaySim, Sparkov. Set aside after measurement (see `docs/data-fit.md`): IEEE-CIS and PaySim have no usable account histories; Sparkov is a second card simulator and adds little variety.
- Limitation: two synthetic generators, both from IBM. Patterns specific to those generators may not transfer.

**M5. AMLWorld: each transfer becomes two account events (sender, receiver); a self-transfer becomes one.**
- Why: an investigator looks at an account, and a transfer is part of both accounts' stories. Counting a self-transfer twice would double its weight in that account's history.
- Considered: scoring transfers only, as the paper does. Kept as a secondary number (M11), not the headline, because analysts review accounts.
- Limitation: both accounts of a laundering transfer are labeled positive, even if one is an unwitting party; the dataset does not say which side is "the launderer".

**M6. AMLWorld: drop everything from 2022-09-11 on.**
- Why: after that date the file is about 60% laundering against about 0.1% before, a generator artifact (laundering patterns finishing after normal activity stops). A test period there would reward any model that learns "late date means laundering".
- Cost: 1,108 of 5,078,345 transfers are dropped, but they include 657 of 5,179 laundering transfers (13%). The usable data is ten days.
- Limitation: our numbers are not directly comparable with work that kept those rows.

**M7. AMLWorld: convert amounts to US dollars with first-day exchange rates.**
- Why: amounts come in 15 currencies; comparing 100 yen with 100 dollars would make "amount" meaningless. Rates are the median of amount received over amount paid in cross-currency transfers. Rates are stable in this generator (the middle 90% of observations sit within 0.1% of the median for every currency except Bitcoin). Only the first day is used, because it is inside every training window, so no later data leaks in.
- Limitation: Bitcoin's rate varies (10,000 to 15,000 USD per BTC in the data), so Bitcoin amounts are approximate.

**M8. AMLWorld comes from a Hugging Face mirror for now.**
- Why: it needs no login. Its row and account counts match the published table exactly (5,078,345 transfers, 515,088 accounts). The official Kaggle copy is free and its checksum should be compared before final results.
- Limitation: until that comparison is done, the data source is a mirror. Recorded in `data/checksums.sha256`.

**M9. TabFormer: one account is one card ("User_Card"); the raw row number breaks timestamp ties.**
- Why: a stolen card is the unit that changes behavior. Timestamps are only to the minute, so a stable tie-breaker is needed for a reproducible order.
- Limitation: TabFormer has 2,000 users holding 6,139 cards; fraud on two cards of the same user is treated as independent.

## 3. Unit of analysis and the review budget

**M10. The unit an analyst reviews is an account-day case.**
- What: one case per account per day it was active. Its score is the highest transaction score that day; it is positive if any transaction that day is labeled.
- Why: analysts open an account and look at its activity, not single rows. One case per day matches a daily queue. "Max" means one very suspicious transaction is enough to put the account in the queue.
- Considered: transaction-level scoring (simpler, but one burst of 10 fraud rows would count as 10 catches); account-level over the whole test period (loses timing).
- Limitation: an account with fraud on several days counts once per day; "first-fraud-day" (M12) corrects for that in the headline.

**M11. The budget is a share of each day's cases: 0.1%, 0.5%, 1%, 2%, 5%.**
- Why per day: a review team has a daily capacity. A pooled budget would let one busy day use slots that belong to another day.
- Why these levels: 0.1% was added because at 1% the card model already catches almost everything, so differences there are invisible. The range shows how results change with capacity.
- Ties: if cases tie at the cutoff, the remaining slots are shared evenly among them (the expected result of a random tie-break), so a model cannot gain or lose by tie order. Fractional slots count proportionally (1% of 250 cases is 2.5 slots).
- Limitation: real teams also prioritize by amount at risk and customer value; a share-of-cases budget ignores that.

**M12. Headline metric: first-fraud-day catch rate, at a 0.5% daily budget on AMLWorld and 0.1% on TabFormer.**
- What: of the accounts that go bad for the first time in the evaluation period, the share whose first bad day lands inside that day's review budget. Accounts already labeled bad before the evaluation window are left out of this metric (they still count in plain recall): catching a known bad account again is not a new catch. On the validation periods, that excluded 161 of 833 AMLWorld accounts (19%) and 477 of 1,033 TabFormer cards (46%); the first version of the metric counted them, which overstated what it measured.
- Why different budgets: at 0.5% the bank-style model already catches about 91% of TabFormer fraud cases, so a gain would be invisible there; 0.1% is the tightest budget available. On AMLWorld it catches about 37% at 0.5%, so 0.5% leaves room. Both were chosen on validation data, before any TSFM existed and before the lock.
- Why first-fraud-day: catching an account the first day it goes bad is what stops further losses; catching it on day five, after four days of damage, is worth much less. It also avoids counting one long burst many times.
- Why 0.5% on AMLWorld: it gives about 1,120 review slots on the validation day (224,616 cases) for 833 laundering accounts, so catching all of them is possible in principle, and the bank-style model is far from that.
- Secondary numbers (always reported, never hidden): recall and precision at every budget, case-level and transaction-level PR-AUC, calibration, cost.
- Limitation: a single headline number simplifies; a model could win the headline and lose elsewhere. All secondary numbers are therefore reported next to it.

## 4. Splits

**M13. Time-ordered windows with a gap: train, then validation, then test.**
- Why: a model is used on the future, so it must be graded on the future. The gap stands in for the delay before fraud labels are known: at the moment of scoring, the most recent labels do not exist yet.
- Validation is used to check the pipeline and choose settings; the test period is scored once, after the lock (Day 9).
- Implemented in `cleanroom.split`, which refuses overlapping or too-close windows and re-checks the actual rows.

**M14. TabFormer split: train to 2015-09, validation 2016, test 2017-04 to 2019, 90-day gaps.**
- Why multi-year test: fraud cards per year are 36 (2017), 269 (2018), and 244 (2019). One year alone could rest on 36 cards. Three years give about 550, and results are also reported per year so a bad year cannot hide inside the average.
- Why 90 days: an assumption for how long fraud reports take. Not sourced to a specific card network rule.
- Why stop at 2019: the file's 2020 data (January and February) contains no fraud.
- Limitation: a model trained through 2015 scoring 2019 is staler than a real bank's model, which would be retrained. A rolling-retrain variant is possible with the same code.

**M15. AMLWorld split: train Sep 1 to 4, validation Sep 6, test Sep 8 to 10, one-day gaps.**
- Why: only ten usable days exist (M6). Training includes a weekend so the model sees both weekday and weekend patterns; the test period covers two weekdays and one weekend day.
- Limitation: a one-day gap is much shorter than real investigation delays; it is what the data allows. Validation is a single day.

**M16. The same account may appear in training and test.**
- Why: a bank scores its existing customers. The split controls when a row is evaluated, not who.
- Limitation: some of a model's skill can come from remembering specific accounts; features are behavior summaries, not account ids, which limits this.

## 5. Features

**M17. Past-only rule: a transaction may use facts about itself and the account's transactions with strictly earlier timestamps. Same-timestamp transactions never see each other.**
- Why: at scoring time, later transactions do not exist, and transactions in the same minute arrive in an unknown order. Treating them as simultaneous is the safe choice.
- Proof: `cleanroom.pastonly` rebuilds features from a copy of the data cut at each probe's timestamp and requires an exact match; it also shuffles the labels and requires no change. A planted bug (windows including their own timestamp) was caught.

**M18. Checklist feature families, and why each exists.**
- Transaction facts (amount, hour, weekday, channel, error flag, refund flag, direction): the classic red flags; known at scoring time.
- Activity counts over 1 hour, 24 hours, 7 days, 30 days: bursts of activity are the most common fraud tell.
- Amount relative to the account's 30-day mean and spread: "unusually large for this customer" matters more than "large".
- Time since the previous transaction: rapid-fire use after a quiet period.
- New merchant or counterparty, new category: stolen credentials are used at places the owner never goes.
- Money in versus out over 24 hours and 7 days, and number of new counterparties: money-laundering red flags (pass-through accounts, fan-in, fan-out). Added so contestant B is a fair opponent on AMLWorld, not a card-fraud model applied to laundering data.
- Limitation: no graph features (the paper's strongest baselines use the whole transaction network). A graph-aware baseline would be stronger; that is out of scope and stated in results.

**M19. Features are computed once on the full history, then split.**
- Why: a test-period transaction should see its own earlier history, including months that fall in the training period. Past-only construction (M17) makes this safe.

## 6. Contestants

**M20. Contestant A ranks by amount, with no learning.**
- Why: the walker in a race. If a model cannot beat "look at the biggest transactions first", it is not worth its cost.

**M21. Contestant B is LightGBM on the checklist features plus category and channel.**
- Why LightGBM: gradient-boosted trees are the standard tool for tabular fraud models, handle missing values and categories natively, and train on 16 million rows on a laptop CPU in about five minutes.
- Considered: XGBoost (similar quality, slower on CPU here), logistic regression (too weak a baseline to be a fair "what banks use").
- Fixed settings, chosen once and not tuned: 300 trees, learning rate 0.05, 31 leaves, at least 200 rows per leaf, 80% row and feature sampling, L2 penalty 1. No early stopping, because early stopping on the validation set would pick the model with the data it is graded on.
- Categories seen fewer than 50 times in training are treated as missing, so rare values do not get memorized.
- Deterministic training with a fixed seed: the same inputs give identical scores.
- Limitation: untuned settings may undersell B a little. The same settings will be used for contestant D, so the comparison stays fair.

**M22. Contestants C (TSFM only) and D (B plus TSFM embedding and surprise score)** are built in Days 12 to 17. Contestant D uses exactly B's settings so any gain comes from the new columns, not from different tuning.

**M23. Where the TSFM is expected to make a difference, stated before testing.**
- Pre-specified hypotheses: (1) the gain, if any, is larger on AMLWorld than on TabFormer, because B is further from the ceiling there; (2) within each dataset, results are split by history length (fewer than 16 prior events versus 16 or more; 16 is the patch length Chronos-2 reads), because a pretrained model may help most where the bank has least data, or may need history to work at all.
- Why state them now: hypotheses chosen after seeing results are easy to fool yourself with.

## 7. Calibration, provenance, and engineering

**M24. Calibration is measured at transaction level, with quantile bins and a base-rate reference.**
- Why transaction level: that is where the model's probabilities live; a case score is a maximum, not a probability.
- Why quantile bins: almost every prediction sits near zero, so equal-width bins would put nearly everything in the first bin.
- Why the base-rate Brier: it shows whether the model beats knowing nothing but the average rate.

**M25. Every run records provenance and is logged forever.**
- Config hash, git commit and whether code had uncommitted changes, data checksum, library versions, fit and score time. Every run appends a line to `artifacts/runs/index.jsonl`.
- Why: results must be traceable to exact inputs, and runs that did not help must stay on record (CLAUDE.md rule 10).

**M26. Features are cached by data checksum plus feature-code checksum.**
- Why: features for 24 million transactions take about two minutes; caching makes iteration fast, and keying by code checksum means an edited feature can never reuse stale values.

**M27. The Cleanroom is one shared package, released by git tag.**
- Why: the sibling projects Reserve and Culprit must sit the same exam. Copies would drift. A test fails the build if the Cleanroom mentions any domain or imports project code.

**M28. CPU first.**
- Why: the owner has no GPU, and every required run must be reproducible on a laptop. Full runs take minutes on CPU; a free GPU (Colab, or ASU's Sol cluster with a faculty sponsor) is optional for TSFM embedding on the full data.


## 8. Honesty machinery (lock, luck checks, canary)

**M29. The sealed-envelope lock: test scoring needs a lock for exactly this config.**
- What: `tells lock <config>` writes `locks/<config hash>.lock.json`, committed to git. Scoring the test period refuses to run without a matching lock, any config change needs a new lock, an edited lock file is rejected, and every test run is appended to `locks/<hash>.uses.jsonl`.
- Why: pre-registration. If the plan is fixed and dated before any test score exists, nobody (including the author) can quietly change the split, budget, or headline after seeing results. Counting test runs exposes "try until it works".
- When: the real configs are sealed just before the Day 26 grid, once contestants C and D exist, so one lock covers the whole plan.
- Considered: a password or a hidden test file. Rejected: a hash committed to git is checkable by anyone and needs no trust.
- Limitation: code edits are not blocked by the lock; git history is what makes them visible (audit check 6).

**M30. Only the headline metric is tested formally; everything else is descriptive.**
- What: every contestant is compared with contestant B on the first-fraud-day catch rate at 0.5%, with a paired bootstrap that resamples whole accounts (1,000 resamples in main configs, 200 in development configs). Holm and Benjamini-Hochberg corrections are applied across all comparisons. P-values use the (count + 1) / (n + 1) form, so a finite bootstrap can never claim p = 0.
- Why accounts, not cases: one account's cases on different days are related; resampling single cases would make intervals falsely narrow.
- Why one formal test: every extra test is another chance of a lucky "win". Secondary metrics are shown, with intervals where useful, but they do not count as findings.
- Why Holm and BH both: Holm is strict (no false win anywhere in the family); BH is the common, more lenient view. Reporting both shows how sensitive a claim is.
- How the queue is handled: each day's review queue is formed once, on the real cases, and the bootstrap resamples which accounts are counted. The first version re-ranked every resample; duplicated accounts then competed for the same review slots, which biased every resample downward (on the TabFormer sample, contestant B's resampled mean was 0.32 against a true value of 0.47, and one interval did not even contain its own point estimate). Each result now also reports the bootstrap mean of the difference, so this kind of bias is visible.
- Limitation: the interval reflects uncertainty about which accounts were observed, not how the queue would change with a different population. The percentile bootstrap can also be inaccurate with very few positive accounts; AMLWorld's single validation day is the weakest case.

**M31. Every run plants two canaries on its own data before reporting numbers.**
- What: on 8 accounts of the run's data (half with a positive label), a future-value feature and a future-label feature are added to the honest builder; both must be caught, and the honest builder must pass, or the run stops.
- Why on every run: leak protection is only meaningful if it is shown to work on the data being reported.
- Found by this check: a sample with no positive accounts makes the outcome canary undetectable, so the sample now always includes positive accounts and the Cleanroom refuses one-outcome samples.
- Limitation: 8 accounts and 10 probes is a spot check, not a full proof; the full proofs are the unit tests.

---

## Limitations register

The honest list. Each item says what could be wrong and what was done about it.

| # | Limitation | Effect | What was done |
|---|---|---|---|
| L1 | Synthetic data only | Results do not show real-bank performance | Stated everywhere; two different generators |
| L2 | Both datasets come from IBM generators | Generator quirks may drive results | Measured fit, found and removed the AMLWorld tail artifact; time-shuffled control planned (Day 18) |
| L3 | AMLWorld usable span is ten days | One-day gaps, one-day validation, short histories | Stated; intervals will be wide and are reported |
| L4 | AMLWorld data from a mirror | Source not yet confirmed against Kaggle | Row counts match the paper; checksum comparison planned |
| L5 | TabFormer fraud is easy for the checklist model | Ceiling effect hides gains at loose budgets | Tighter 0.1% budget; first-fraud-day headline; AMLWorld as main test |
| L6 | 90-day and 1-day gaps are assumptions | Real label delays differ | Stated; gap is a config value |
| L7 | No graph features in contestant B | B is weaker than the paper's best baselines on AMLWorld | Stated; money-movement features added so B is not a card-only model |
| L8 | Contestant settings untuned | B may be slightly undersold | D uses identical settings, so the comparison is fair |
| L9 | Same accounts in train and test | Some skill may be account memory | Features describe behavior, not identity |
| L10 | Both sides of a laundering transfer are labeled | Unwitting counterparties count as positives | Dataset limitation; stated |
| L11 | Headline chosen on validation data | Choosing the budget could favor some model | Chosen before any TSFM existed and before the lock; all budgets reported |
| L12 | Intervals on few positives | AMLWorld validation is one day; intervals can be wide or unstable | Account-level paired bootstrap, (count + 1) / (n + 1) p-values, Holm and BH, all reported |
| L13 | The lock does not block code edits | A person could change code between lock and test | Git history shows every edit; the agent will only reach data through the locked harness (Day 21 to 25) |
| L14 | Canary is a spot check | 8 accounts, 10 probes per run | Full past-only proofs are in the unit tests; the canary proves the checks work on the run's own data |
