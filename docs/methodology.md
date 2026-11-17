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
- Why AMLWorld is the main test: (1) contestant B, the bank-style model, is far from perfect on it (on the full validation day it caught 31.5% of newly bad accounts on their first day at a 0.5% budget, against about 96% of card fraud cases on TabFormer), so a TSFM gain can actually show; (2) it is the benchmark the published TSFM paper used, so results can be set next to theirs; (3) its accounts have short histories (median 9 events before the first laundering), which tests whether a pretrained model helps where a bank has little data on a customer.
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
- Why different budgets: at 0.5% the bank-style model already catches about 91% of TabFormer fraud cases, so a gain would be invisible there; 0.1% is the tightest budget available. On AMLWorld it catches 31.5% of newly bad accounts on their first day at 0.5% (validation), so 0.5% leaves room.5% leaves room. Both were chosen on validation data, before any TSFM existed and before the lock.
- Why first-fraud-day: catching an account the first day it goes bad is what stops further losses; catching it on day five, after four days of damage, is worth much less. It also avoids counting one long burst many times.
- Why 0.5% on AMLWorld: it gives about 1,120 review slots on the validation day (224,616 cases) for 833 laundering accounts, so catching all of them is possible in principle, and the bank-style model is far from that.
- Secondary numbers (always reported, never hidden): recall and precision at every budget, case-level and transaction-level PR-AUC, calibration, cost.
- Limitation: a single headline number simplifies; a model could win the headline and lose elsewhere. All secondary numbers are therefore reported next to it.

## 4. Splits

**M13. Time-ordered windows with a gap: train, then validation, then test.**
- Why: a model is used on the future, so it must be graded on the future. The gap stands in for the delay before fraud labels are known: at the moment of scoring, the most recent labels do not exist yet.
- Validation is used to check the pipeline and choose settings; the test period is scored once, after the lock (built on Day 9, used on Day 26).
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
- Changed on 2026-11-16: test scoring now also refuses when the code has uncommitted changes or the commit is unknown, and when the lock file is not tracked by git, because an uncommitted lock is not dated by anything. Each test run writes a "started" line to the uses file before test labels are revealed and a "finished" line after, and the test-run number counts started runs, so a run that crashes still counts. Older lines without a status count as one run each. `tells lock` now also pins the data file's checksum; scoring refuses different data under that lock. Locks sealed before this (without the checksum) keep working. Unit tests skip the git checks through an explicit `allow_untracked_lock_for_tests` argument, never the lock itself. Side effect: because the uses file is tracked, it must be committed after one test run before the next can start.

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
- Changed on 2026-11-16: the canary now uses only rows before the test window starts. It picks half its accounts by label, and before this it read labels from the whole history, so test-period labels shaped which accounts it chose. No score came from this, but the rule is that nothing reads test labels before the lock.


## 9. TSFM inputs and compute

**M32. One TSFM embedding per account-day, from the history before that day began; Chronos-2 only where a CPU can finish.**
- What: for every account-day case, the account's events strictly before midnight of that day form one sequence; its embedding is shared by all of that day's transactions. Embeddings are cached by model, version, account, and cutoff. The original plan was to run Chronos-2 on full data once on a free GPU; no GPU was available, so Chronos-2 ran only on the 20,000-account AMLWorld sample and TTM on full data (M36).
- Why per account-day: one embedding per transaction (millions) is unaffordable, and the unit being ranked is the account-day anyway. Cutting at midnight is stricter than past-only per transaction, so it cannot leak.
- Why a GPU for this step only: measured on this laptop, Chronos-2 takes about 0.4 seconds per account at 256 events and 4 input series, so the AMLWorld validation day alone would need about a day of CPU. A GPU is typically one to two orders of magnitude faster for this kind of model (not measured here yet).
- Considered: embedding once per account per window (cheaper, but stale within the window and empty for AMLWorld training accounts, whose data starts on day one); subsampling accounts for contestant D only (rejected: B and D must be trained on identical rows for a fair comparison).
- Limitation: same-day transactions before the scored one are invisible to the embedding (the checklist still sees them). The embedding run depends on an external notebook environment; its inputs and outputs are checksummed so the cache can be verified.
- Changed on 2026-11-16: the cache folder name was kept (so finished multi-hour caches stay valid), but `meta.json` now also records a checksum of the requested account-days, the chunk size, and a checksum of the adapter code. Chunks are resumed by position, so resuming with a different key list or chunk size could duplicate or drop account-days; that is now refused with a message to delete the cache. Fields missing from an older `meta.json` are treated as unknown and allowed. Loading the cache also checks that each account-day appears once and that every requested account-day is present.


**M33. Surprise score: how far a transaction is from what the model expected.**
- What: from the history before midnight, each TSFM forecasts the account's next event: its log amount and its log minutes since the previous event. For every transaction that day, surprise = actual minus forecast, for amount and for gap. Two columns per model.
- Why: a stolen card or mule account often breaks its own rhythm (a much larger amount, or activity far sooner than usual). The embedding summarizes the history; the surprise score says how unusual today is against it. Its inputs are past-only (the forecast) and the transaction's own amount and time since the previous event, both known when it is scored.
- Considered: forecasting the whole day's activity, or using the full quantile spread. Kept to a median forecast to stay simple and cheap.
- Limitation: there is one forecast per account-day (the next event after midnight), so for the third transaction of a day the forecast is still "the first event of the day"; the checklist's same-day counts cover what happens within the day.

**M34. From embeddings to tree-model columns: average the history patches, then PCA to 16 components fitted on training days only.**
- What: Chronos-2 gives 768 numbers per 16-event patch for each of the 3 channels; TTM gives 192. Patches are averaged per channel (as in Savvidi et al. 2026) and concatenated (2,304 numbers for Chronos-2, 576 for TTM). PCA is then fitted on the training-window account-days only, with no labels, and the first 16 components are kept. Contestant C uses only the TSFM columns (16 components, 2 surprise scores, a has-history flag); contestant D uses B's columns plus those, with B's exact LightGBM settings.
- Why average: it gives a fixed-length summary regardless of how long the history is. Why PCA: hundreds of strongly correlated numbers make tree models slow and prone to noise; 16 components keep most of the variation (the share kept is recorded in every run as `pca_variance_explained`). Why training days only: anything fitted on validation or test data would let those periods shape the inputs.
- Considered: the [REG] summary token (Chronos-2 only, so not comparable across models); keeping all dimensions (too slow on CPU; planned as a Day 26 ablation, not run: each ablation needs a new multi-hour embedding run on CPU).
- Changed on 2026-11-14: embedding numbers are now standardized (divided by their training-period spread) before PCA. Without it, a few TTM numbers with spreads up to 479 (median 7) took 97% of the variance, so the "16 components" were really about two directions, one of them mostly history length. The unstandardized run stays on record as a tried variant (M41).
- Limitation: averaging loses the order of patches inside the history, and 16 components may drop rare patterns; both are ablation items.

**M35. TTM reads a fixed 512-step window, so short histories are padded and marked as not observed.**
- Why: TTM r2 only accepts its trained context length. Padding with zeros alone would tell the model "many free, instant transactions"; the `past_observed_mask` tells it those steps are not real data.
- Limitation: how much the padding still influences TTM's summary is not measured; AMLWorld histories are mostly short (median 10 events in a 512 window).

**M36. What runs where, given no GPU.**
- Measured CPU cost per account-day: TTM about 2.3 to 2.9 ms; Chronos-2 about 66 ms on AMLWorld (short histories) and 283 ms on TabFormer (256 events), plus a similar cost again for its forecast.
- So: TTM runs on the full AMLWorld main test and on the TabFormer sample; Chronos-2 runs on the 20,000-account AMLWorld sample only. Full-data Chronos-2 is left out (no GPU). Full TabFormer with TTM (about 7 CPU hours) was first left out, then sealed and run at the owner's request (lock `701e5f22`).
- Why this is still a fair test: contestant D is compared with B on exactly the same rows each time, so a sample result is a valid comparison, just with wider intervals.
- Limitation: the Chronos-2 result rests on a 20,000-account sample (about 23 newly bad accounts on its validation day), so its interval will be wide and cannot confirm a small effect.


## 10. Controls and explanations

**M37. Label-permutation control: a model trained on noise must score like random ranking.**
- What: the reference model (B) is retrained on training labels shuffled within each day, keeping each day's fraud rate but breaking any link between behavior and fraud. It is scored on the evaluation period and compared, with the paired account bootstrap, to a constant score (random ranking, which catches exactly the budget share).
- Why: if a noise-trained model beats random, labels are reaching the evaluation some other way (a leaky feature, a cache, a join), and every other result would be suspect. It complements the canary: the canary tests the feature checks, this tests the whole pipeline end to end.
- Considered: shuffling the order of each account's history ("time-shuffled control"). Kept for the TSFM ablations (does order matter?), because it answers a science question rather than a leak question, and it needs a second embedding run.
- Limitation: it detects leaks that carry label information; it cannot detect a leak that only makes the real model look slightly better.
- See M42 for how far the shuffle reaches (within each day, not across days) and why that mattered on TabFormer validation.

**M38. Explanations use LightGBM's built-in SHAP values.**
- What: on up to 20,000 evaluated transactions, each column's contribution to each score is computed (TreeSHAP), averaged in absolute value, and the share coming from TSFM columns is reported. The five highest-scored newly bad account-days are listed with their three biggest reasons.
- Why: a fraud team must be able to say why an account was flagged. The TSFM share answers a central question directly: when D beats B, is it because of the TSFM columns?
- Considered: permutation importance (slower, and unstable with correlated columns); the separate `shap` package (an extra dependency for the same TreeSHAP values).
- Limitation: SHAP explains the model, not the world; an embedding component is still hard to put into words. Example accounts are synthetic.


## 11. Reporting and the agent

**M39. Reports are rendered by code from run files; the agent gets three tools and nothing else.**
- What: `tells report <run>` builds every table (headline with intervals, every budget, cost, the TSFM share of attribution, the label-permutation control, the history-length split) from `results.json` and the saved case files. The agent's tool layer allows only `run_experiment` (configs by name, test period only through the lock), `load_results` (summary numbers, never rows), and `plot` (chart data). Arguments are checked against allow-lists and strict patterns, and every call is logged.
- Why: CLAUDE.md rule 4 (numbers come from result files) is only enforceable if no person or model types a number. Limiting the agent to three narrow tools means it cannot read test labels, raw data, or anything outside the project, whatever it is asked.
- Considered: giving the agent a Python sandbox (far more flexible, far harder to audit).
- Limitation: the history-length split for runs made before it existed is recomputed from saved cases and data; the recomputation is tested to use the same functions as live runs.
- Changed on 2026-11-16: the report never writes the feature cache. It reads the run's cache if one exists for that data and feature code, and otherwise builds the features in memory from every column. Before, it read a subset of columns and could save an incomplete cache (without the transfer-direction and error columns) that a later run would then reuse. It also refuses if the data file's checksum differs from the one recorded in the run.
- Changed on 2026-11-16: the agent can no longer request the test period by default. `run_experiment` offers only `valid` unless `tells agent --allow-test` is given, and a `test` request is denied before it reaches the harness. Why: spending the one sealed test run is a person's decision, not something a model should choose while exploring.


**M40. The agent: Claude Opus 5.5 in a loop we own, a hard tool budget, and a Critic that checks every claim.**
- What: `tells agent "<question>"` runs Claude Opus 5.5 (`claude-opus-5-5`, effort `high`, strict tool schemas, server-side refusal fallback) in a manual loop: every tool call goes through the three-tool allow-list, denials go back to the model as errors, and the loop stops after 10 tool calls. The model must not write numbers; it ends with a JSON list of the runs it relied on and its claims (contestant, metric, run, direction). The Critic then checks, without any model: no typed numbers, every run exists and passed its canary and control, test runs were locked, and every claimed direction matches the run's bootstrap interval. The Reporter puts the model's narrative next to tables generated from the result files and labels the document a draft for human review.
- Why a manual loop instead of the SDK's tool runner: every call must pass our gatekeeper and every exchange must be recordable as a cassette, so the loop is ours. Why Opus 5.5: it is the current default Claude model; planning experiments rewards care more than speed. Why the refusal fallback: if the model declines a request, the API retries on a fallback model instead of failing the run (on by default; can be turned off in `src/tells/agent/llm.py`).
- Tests: a hand-written cassette (not a recording of the real API) exercises the loop, the denials, the budget, the Critic, and the Reporter; replay refuses a request that differs from the recording. Recorded cassettes need an API key.
- Limitation: the Critic checks claims against intervals, not the quality of the reasoning; an LLM review pass is planned once a key is available. The agent costs money per question (Opus 5.5 pricing in the Claude API docs).
- Changed on 2026-11-16: four Critic fixes. The narrative now ends where the JSON summary line starts (cutting at the last "{" landed inside the nested claims). The typed-number check removes only exact known identifiers (run ids, contestant and feature names from the runs relied on, config names), ignoring case, and also rejects number words such as "two", "half", "doubled", and "percent"; before, only lowercase names were removed, so an honest answer naming D_checklist_tsfm_chronos2 failed. A claim's metric must be the metric the run actually tested. A direction of "better" or "worse" now needs the interval to exclude zero and a Holm-corrected p-value below 0.05, so claims account for how many contestants were compared (CLAUDE.md rule 11); otherwise the only accepted claim is "no clear difference". Limitation: the number-word list is fixed, so an unusual phrasing ("a dozen") can still slip through.


**M41. On AMLWorld, accounts in training have far less history than accounts in evaluation; every run now reports this.**
- Measured: account-days with 16 or more earlier events are 9.3% of training (Sept 1 to 4, median 1 event), 48.4% of validation (Sept 6, median 15), and 62.7% of test (Sept 8 to 10, median 20). The data begins on Sept 1, so histories can only grow.
- Why it matters: a model learns what TSFM summaries of very short histories look like, then is graded mostly on longer ones it has rarely seen. In the first TTM run (unstandardized PCA), contestant D matched B on short-history accounts (27.9% against 28.1%) but collapsed on long-history ones (3.2% against 38.7%), which pulled its headline 11.6 points below B. The label-permutation control was consistent with chance, so this is not a leak.
- What was done: standardized PCA (M34), and `history_mix_share_16_plus` in every run's results, so the shift is visible next to the numbers.
- Variants tried so far on the headline (all counted for multiple-comparison purposes): TTM with unstandardized PCA (run `20261115T031830Z_amlworld_valid_f6ff26be_96919d`); TTM with standardized PCA (run `20261115T041547Z_amlworld_valid_f6ff26be_5e9b71`), which is the variant that was sealed and tested.
- Limitation: this is a property of AMLWorld's ten-day span, and no training-window choice removes it. Results on AMLWorld must be read as "TSFM features learned on new accounts, applied to established ones". TabFormer, with decades of history in training, does not have this problem.


## 12. After the sealed tests

**M42. Shuffle scope of the label-permutation control.**
- What: the built-in control (M37) shuffles training labels within each day. On TabFormer validation (run `20261116T193405Z_tabformer_valid_701e5f22_d9a420`) the noise-trained model caught 5.9% of newly bad cards on their first bad day at the 0.1% budget, against 0.1% for random ranking (difference +5.8 points, 95% interval +4.0 to +7.8), so the control was not consistent with chance. On the sealed TabFormer test it was (+0.4 points, -0.1 to +1.4).
- Why it can happen without a leak: shuffling within a day keeps each day's fraud count, so the noise-trained model still learns which days had fraud. TabFormer fraud comes in bursts on particular days, and day-level traits (calendar, how busy the day is) can then carry real signal into the shuffled model. A stricter control shuffles labels across all training days, which removes that day structure. It was run as a descriptive follow-up, not a pre-registered test: `scripts/control_shuffle_scope.py` retrains the reference model with within-day, across-all-days, and a second across-all-days seed, and writes `docs/analysis/tabformer-shuffle-scope-control.md`.
- Conclusion: SHUFFLE_SCOPE_PENDING
- Considered: switching the built-in control to across-all-days shuffling. Not done for the sealed runs, because the control is part of the locked plan; changing it after seeing a result would be the kind of edit the lock exists to prevent.
- Limitation: within-day shuffling is a weaker null on bursty data; a control that beats random under it is ambiguous until the stricter shuffle is checked.

**M43. Model staleness on TabFormer.**
- What: contestant B is trained through 2015 (M14) and never retrained. Its first-fraud-day catch rate at the 0.5% budget fell from 90.1% on the 2016 validation year to 29.9% on the 2017 to 2019 test period (at the 0.1% headline: 28.4% to 11.2%).
- Why the split was still right: it was pre-registered and sealed, and the test needs several years to reach enough newly bad cards (M14: one year alone could rest on 36 cards). Changing it after seeing validation would have broken the lock's promise.
- Considered: a rolling retrain (retrain before each test year). Possible with the same code, but it would be a new plan with a new lock, not a repair of this one.
- Limitation: the TabFormer test numbers describe a stale model scoring years ahead, which is not how a bank runs. Fraud patterns in 2017 to 2019 may also differ from 2016, so staleness is the likely cause of the drop, not a measured one. D was trained on the same window, so the B versus D comparison stays fair.

**M44. Descriptive analyses after sealing.**
- What: `scripts/analyze_test_run.py` reads only a finished run's saved case files and `results.json`, computes every budget with 95% intervals (account bootstrap, 300 resamples) and other breakdowns, and writes a Markdown file whose numbers are formatted by code. It applies no multiple-comparison correction and labels every number "descriptive, not pre-registered". It was run only after both sealed tests were scored.
- Why: the sealed tests answer one pre-registered question each. Readers will still ask about other budgets and subgroups; answering from saved files, clearly labeled, is better than leaving the questions open or quietly promoting a secondary number.
- Considered: 1,000 resamples as in the main configs (slower, and the extra precision does not matter for numbers that are not tests); correcting across all breakdowns (would suggest these are tests).
- Limitation: these numbers were chosen after seeing results, so any pattern in them is a hypothesis for a future sealed plan, not a finding.

---

## Limitations register

The honest list. Each item says what could be wrong and what was done about it.

| # | Limitation | Effect | What was done |
|---|---|---|---|
| L1 | Synthetic data only | Results do not show real-bank performance | Stated everywhere; two different generators |
| L2 | Both datasets come from IBM generators | Generator quirks may drive results | Measured fit, found and removed the AMLWorld tail artifact; label-permutation control on every run (M37); a time-shuffled variant was deferred |
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
| L13 | The lock does not block code edits | A person could change code between lock and test | Git history shows every edit; the agent reaches data only through the three-tool layer and the locked harness (built, Days 21 to 25). The lock hashes the config file, not the code: settings that live in code (for example PCA standardization) are pinned by the git commit stamped on each run |
| L14 | Canary is a spot check | 8 accounts, 10 probes per run | Full past-only proofs are in the unit tests; the canary proves the checks work on the run's own data |
| L15 | Full Chronos-2 embedding needs a GPU | Main TSFM results depend on one notebook run | Inputs and outputs checksummed; cache keyed by model, version, account, cutoff; CPU path for samples and TTM |
| L16 | Chronos-2 only on a sample | Its result has wide intervals | Same rows for B and D; TTM on full data; GPU run left as future work |
| L17 | One forecast per account-day | Later same-day transactions compare with the first-event forecast | Checklist same-day counts cover within-day activity |
| L18 | AMLWorld history shift between training and evaluation | TSFM summaries in evaluation look unlike those in training | Measured and reported per run (M41); standardized PCA; TabFormer as the counter-case |
| L19 | The label-permutation control shuffles within each day | On bursty data a noise-trained model can beat random without a leak, so a failed control is ambiguous | Stricter across-all-days shuffle run as a descriptive follow-up (M42); the sealed TabFormer test's control was consistent with chance |
| L20 | TabFormer model trained through 2015 scores 2017 to 2019 without retraining | Test numbers reflect a stale model; B's 0.5% catch fell from 90.1% to 29.9% | Split kept because it was pre-registered (M14, M43); same window for B and D; rolling retrain left as a new sealed plan |
