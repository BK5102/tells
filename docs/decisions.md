# Decisions

One entry per decision or feature: what was decided, why, and an analogy. These are the owner's talking points. Newest at the bottom.

## Phase 0 (2026-11-10)

### D1. One Cleanroom package, shared by all three repos

The Cleanroom is written once, as its own installable Python package inside the Tells repo (`packages/cleanroom/`), and Reserve and Culprit install a tagged version of it from git instead of keeping their own copies. Copies drift: a leak bug fixed in one repo would stay alive in the other two, and the claim "all three projects sit the same honest exam" would stop being true. Analogy: three schools using the same sealed exam board, rather than each school photocopying the rulebook and editing its own version.

How siblings install it (after the `cleanroom-v0.1` tag exists):

```
uv add "cleanroom @ git+https://github.com/BK5102/tells@cleanroom-v0.1#subdirectory=packages/cleanroom"
```

Rules that follow from this: the Cleanroom contains no fraud, credit, or payments words; any change is released as a new tag (`cleanroom-v0.2`, ...); siblings upgrade on purpose, never automatically. If the Cleanroom later outgrows the Tells repo, it can move to its own repo with no change to the import name.

### D2. Datasets: TabFormer first, AMLWorld second, PaySim dropped

Measured fit is in `docs/data-fit.md`. TabFormer cards have a median of 1,493 transactions before their first fraud across 29 years, so it is the dataset where "read the account's rhythm" can actually be tested; AMLWorld HI-Small is kept as the second dataset because it is the benchmark the published TSFM paper used, even though its accounts have a median of only 9 prior events. PaySim was dropped because 99.9% of its accounts appear exactly once, and Sparkov is an optional extra once a free Kaggle account exists. Analogy: to test whether a bartender can spot a regular acting strangely, you need a bar full of regulars, not a train station.

Update (2026-11-10): IEEE-CIS-only was considered, because it is real data and avoids the TabFormer and AMLWorld quirks. It was rejected after measuring it: with the standard account reconstruction the median account appears once and has zero transactions before its first fraud, and 66% of transactions after an account's first fraud are also labeled fraud, so labels were copied forward. The TabFormer and AMLWorld problems cost one config line each to handle, while IEEE-CIS's missing histories cannot be fixed. Full ranking in `docs/data-fit.md`.

Final (2026-11-10): Tells uses only TabFormer and AMLWorld HI-Small, both free. Dropping money laundering and using Sparkov instead was considered, but two card-fraud simulators would be less varied than one card-fraud dataset with long histories plus one laundering dataset with short histories and a published TSFM baseline. Analogy: testing a new running shoe on both a track and a trail tells you more than testing it on two tracks.

### D3. Scoring unit: account-day cases, with transaction PR-AUC as a secondary number

The review budget is applied to account-day cases: each account active on a day is one case, its score is the highest score among its transactions that day, and it counts as fraud if any of those transactions is labeled. Analysts review accounts, not single rows, and both datasets can be expressed this way, while transaction-level PR-AUC is also reported so results can be compared with the published AMLWorld numbers. Analogy: a security guard checks each traveler once per day at the gate, not every bag they touch.

The budget is a share of each day's cases (for example the top 1% of today's cases), so a busy day does not crowd out a quiet day. For TabFormer, also report "caught on first fraud day", because fraud comes in bursts and the first catch is what saves money.

### D4. CPU first, GPU optional

Every required run must finish on a laptop CPU: the small config samples accounts, embeddings are cached on disk, and TSFM choice on Day 12 favors models that run on CPU. A free Colab GPU may be used for the full grid on Days 26 and 27, recorded in the run log. Analogy: write the recipe so it works in a home kitchen; a restaurant oven only makes it faster.

The owner has no physical GPU. Options for the full grid, in order: ASU Research Computing's Sol supercomputer (free GPU nodes for ASU research, but a student account needs a faculty sponsor; request through the Research Computing account form, see [docs.rc.asu.edu/accounts](https://docs.rc.asu.edu/accounts)), then free Google Colab or Kaggle notebook GPUs. Nothing in the 30-day plan requires a GPU.

### D5. Cleanroom v0.1 includes the canary and multiple-comparison correction

The original plan built the leak canary on Day 18 and had no day for multiple-comparison correction, but START_HERE promises the canary in the Cleanroom and Reserve requires the correction. Both now ship in Days 10 and 11, before the Cleanroom tag, and Day 18 only wires them into Tells experiments. Analogy: the smoke detector is installed when the house is built, not after the first dinner party.

## Day 1 (2026-11-10): repo scaffold

The repo is a uv workspace with two packages: `tells` at the root and `cleanroom` in `packages/cleanroom`, each with its own `pyproject.toml`, tested together by one `pytest` run and by GitHub Actions on Python 3.11 and 3.12. A test fails the build if the Cleanroom ever imports from `tells` or mentions fraud, cards, loans, or payments, so the "domain-neutral" rule is enforced by code, not by memory. Analogy: two apartments in one building, sharing the front door but each with its own lock and lease.

## Day 2 (2026-11-10): TabFormer loader, small sample, account-day cases

The TabFormer loader turns the raw CSV into the standard schema (one row per transaction, each card's history sorted oldest to newest, the raw row number kept as `tx_id`), and `uv run tells prepare tabformer` saves it as Parquet plus a 300-card sample in 26 seconds; a test confirms the full file matches the counts measured independently in `docs/data-fit.md` (24,386,900 transactions, 6,139 cards, 29,757 frauds). The sample picks cards by hashing their ids with a seed, so it is identical on every machine and a bigger sample always contains the smaller one, and `build_cases` groups transactions into account-day cases with a flag for each card's first fraud day. Analogy: the loader is a clerk who files every receipt into the right customer's folder in date order, and the case builder is the daily stack of folders an investigator could open.

## Day 3 (2026-11-10): `cleanroom.split` and the proposed TabFormer split

`cleanroom.split` defines a split as ordered, half-open time windows (train, optional validation, test) with a required minimum gap between them; it refuses overlapping or too-close windows when the config is read, and `Split.check` re-verifies on the actual rows that every later part starts at least the gap after the earlier part ends, which a test proves by smuggling a test row into train and watching it fail. The proposed TabFormer split in `configs/tabformer.yaml` is train through 2015-09, validation 2016, test 2017-04 through 2019, with 90-day gaps standing in for the delay before fraud labels are known; on the real data that gives 20,033 / 3,579 / 4,833 fraud transactions and passes the check. Analogy: study from last year's papers, take a three-month break so no answers leak by word of mouth, then sit this year's exam.

The same card can appear in train and test. That is intended: a bank scores its existing customers, and the split only controls *when* rows are evaluated. The split becomes binding when the lock lands on Day 9.

## Day 4 (2026-11-10): lazy baseline and the scoring harness

Contestant A ranks transactions by amount, biggest first, with no learning at all, and every contestant shares one interface: learn from training rows, then score rows without seeing their labels. The harness (`uv run tells run configs/small.yaml`) splits through the Cleanroom, re-checks the split on the data, drops the test period's labels the moment the split is made, scores the validation period, groups scores into account-day cases, and saves everything under `artifacts/runs/`; asking for the test period raises an error until the lock exists. Analogy: the walker in a race is there so that a runner's time means something, and the referee keeps the finish-line photo sealed until the race is declared official.

The 300-card sample has only 71 fraud cases in its 2016 validation year, so small-config numbers are for checking the pipeline, not for conclusions.

## Day 5 (2026-11-10): budget metrics and PR-AUC

`cleanroom.metrics` measures what a review team cares about: give each day a fixed number of review slots (for example 1% of that day's cases), fill them from the highest score down, and count the share of all fraud cases caught (recall), the share of reviewed cases that were fraud (precision), and the share of cards caught on their first fraud day. When cases tie at the cutoff, the leftover slots are shared evenly among them, which equals breaking ties at random, so a model cannot gain or lose by tie order; PR-AUC is also computed, at both case and transaction level, and every function is tested against hand-worked answers. Analogy: an airport can only pat down a few passengers per flight, so the question is not "how accurate is the scanner" but "how many smugglers were in the pat-down line".

## Day 6 (2026-11-10): calibration

`cleanroom.calibration` adds the Brier score (average squared gap between predicted probability and what happened), a reliability table that bins predictions and compares each bin's average prediction with its observed fraud rate, and the expected calibration error that summarizes those gaps; bins default to equal-count quantiles because almost every fraud prediction sits near zero and equal-width bins would put nearly everything in one bin. The harness reports calibration only for contestants whose scores are probabilities, at transaction level, next to the Brier score of always predicting the base rate so a reader can tell whether the model beats knowing nothing. Analogy: a weather forecaster who says "70% rain" should see rain on about 7 of every 10 such days, and should do better than one who always quotes the yearly average.

## Day 7 (2026-11-10): past-only checklist features and the leak test that guards them

`build_checklist_features` adds 25 red-flag features per transaction: facts about the transaction itself (amount, hour, online or chip, error flag) and the card's history from strictly earlier timestamps only (counts and amounts over the last hour, day, week, and month, time since the previous transaction, how unusual this amount is for the card, and whether the merchant or category is new to it). The guard is `cleanroom.pastonly`, written first: for sampled transactions it rebuilds the features from a copy of the data containing only earlier rows and demands an exact match, and it shuffles the labels and demands no change; planting a bug that let each window include its own timestamp made two leak tests fail, so the guard is shown to bite. Analogy: to check a witness only describes what they saw before the crime, you replay the tape stopped at that moment and ask again, and the story must not change.

Transactions that share a timestamp are treated as simultaneous and never see each other. The full 24 million transactions take about 2 minutes to featurize; the 300-card sample takes 3 seconds.

## Day 8 (2026-11-10): contestant B (checklist + LightGBM), typed config, run log, provenance

Contestant B feeds the 25 checklist features plus the merchant category into LightGBM with a fixed number of trees and no early stopping, so the validation period is never used to choose the model it is graded on; its settings are a typed config that rejects misspelled keys, and the same seed gives identical scores. Every run now records its provenance (config hash, git commit and whether the code had uncommitted changes, data checksum, library versions), the fit and scoring time, and appends one line to `artifacts/runs/index.jsonl`, so no run, good or bad, disappears; checklist features are cached under a key built from the data checksum and the feature code's checksum, so editing either rebuilds them. Analogy: a lab notebook where every page is dated, says which batch of reagent was used, and can never be torn out.

First pipeline check on the 300-card sample, validation year 2016 only (not a result: 71 fraud cases, no intervals yet, no controls): contestant B caught 72% of fraud cases at a 1% daily budget against 13% for the lazy baseline. A jump that large on synthetic data is a reason to run the time-shuffled control (Day 18) before believing it.

### Day 8 follow-up: full-data validation run and a ceiling problem

Run `20261111T034334Z_tabformer_valid_1ddd8d53` (all 6,139 cards, validation year 2016, 3,579 fraud transactions, no intervals or controls yet): at a 0.5% daily budget the lazy baseline caught 5.4% of fraud cases and contestant B caught 90.6%; at 1%, 12.7% and 95.6%; B's transaction PR-AUC was 0.832 and its Brier score 0.0007 against 0.0021 for always predicting the base rate. Fit took 294 seconds on 16.7 million training rows, scoring 22 seconds for 1.7 million rows. The run was stamped with uncommitted code (`git.dirty = true`) because it started before the Day 8 commit.

This creates a ceiling problem for the main question: if the checklist model already catches about 96% at a 1% budget, a TSFM can add at most a few points there, and a real gain could be invisible. Before Day 9 locks the plan, decide how to keep the comparison informative, for example by adding a tighter budget (0.1% of daily cases), making first-fraud-day catch the headline metric, or leaning on AMLWorld, where the published baselines are far from perfect. The very high score is also a reason to run the time-shuffled control (Day 18) early.

## 2026-11-14: AMLWorld becomes the main test; first-fraud-day is the headline

The full TabFormer run showed the bank-style model already catching about 96% of fraud cases at a 1% budget, which leaves no room to see a TSFM gain. So AMLWorld is now the main test (its loader was pulled forward from Day 27), a tighter 0.1% budget was added, and the headline metric is the share of bad accounts caught on their first bad day at a 0.5% daily budget. To keep the comparison fair, the checklist gained money-movement red flags (money in versus out, bursts of new counterparties), so contestant B is a real anti-money-laundering baseline, not a card model applied to laundering data. Analogy: if every student scores 98% on an easy exam, you cannot tell who studied, so you grade them on the hard paper. Every reason is in `docs/methodology.md` (M4, M5 to M8, M11, M12, M18, M23).

## Day 9 (2026-11-14): the sealed-envelope lock

`uv run tells lock <config>` writes a lock file named after the config's hash; scoring the test period refuses to run unless a lock for exactly that config exists, any config change needs a new lock, an edited lock is rejected, and every test run is counted. The real configs will be sealed before the Day 26 grid, once all four contestants exist, so one lock covers the whole plan. Analogy: a scientist files the experiment plan with a notary before running it, and the notary's stamp shows the date.

## Day 10 (2026-11-14): paired bootstrap and multiple-comparison correction

Each run now compares every contestant with the bank-style model on the headline metric by resampling whole accounts 1,000 times (an account's days belong together), reporting the difference with a 95% interval and a p-value that can never claim to be exactly zero, then applying Holm and Benjamini-Hochberg corrections across comparisons. Only the headline is tested formally, so the number of chances to get lucky stays small. Analogy: a free-throw shooter who makes three in a row may just be having a good night; you watch many nights before calling them better.

## Day 11 (2026-11-14): canary, leak audit, Cleanroom v0.1

Every run now plants two cheats on its own data (a feature that peeks at the next transaction, and one that peeks at the next label) and refuses to report anything unless both are caught and the honest features pass. The canary immediately found a weakness: a sample with no laundering accounts makes the label cheat invisible, so samples now always include positive accounts. The written audit is in `docs/audits/2026-11-14-day11-leak-audit.md`; the Cleanroom is released as version 0.1.0 under the git tag `cleanroom-v0.1`. Analogy: a smoke detector is tested by holding a match to it, not by trusting the label on the box.

## Validation check, 2026-11-14 (contestants A and B, both datasets, final Phase 3 code)

AMLWorld validation day (run `20261115T020923Z_amlworld_valid_5152975d_4396a0`): the bank-style model caught 31.5% of newly bad accounts on their first bad day at a 0.5% budget, against 2.1% for the lazy baseline (difference -29.5 points, 95% interval -33.0 to -25.9). TabFormer 2016 (run `20261115T021607Z_tabformer_valid_44c6db59_66d055`): 28.4% against 0.7% at the 0.1% budget; at 0.5% the bank-style model reaches 90.1%, confirming the ceiling that led to the tighter budget. Both runs passed the canary, and the bootstrap mean sat on the point estimate in both, confirming the bias fix on real data.

## Day 12 (2026-11-14): which TSFMs, and what they cost on a laptop

Chronos-2 (119.5M parameters, Apache-2.0) has an official `embed` method that returns one embedding per 16-step patch and reads several parallel series of one account at once, so it is the first model; TTM r2 (0.8M parameters, a different lab and design) is the second, for contrast. Measured on this laptop, Chronos-2 takes about 0.4 seconds per account at 256 events and 3 to 4 channels, so embedding every account-day of the AMLWorld validation day alone would take about a day on CPU; full Chronos-2 embedding will run once on a free GPU and be cached, and everything else stays on CPU. Analogy: you can grind coffee by hand for one cup, but for a hundred cups you borrow the café's grinder once and store the grounds. Details: `docs/tsfm-notes.md`, methodology M32.

## Day 13 (2026-11-14): account histories as sequences

`build_sequences` turns each account-day case into the account's most recent 256 events from before midnight that day, as three parallel channels: log amount, log minutes since the previous event, and direction (in, out, internal). The past-only check passes, and a planted bug that let each sequence include its own day's events failed two tests. Analogy: the model reads a customer's diary up to last night, never today's page.

## Days 14 to 17 (2026-11-14): TSFM adapters, cache, surprise score, contestants C and D

Every TSFM sits behind one adapter that takes account histories and returns a fixed-length summary (embedding) and a forecast of the next event; Chronos-2 and TTM both have one, and a tiny fake adapter lets every test run without PyTorch. Embeddings are cached on disk in resumable chunks, squeezed to 16 numbers with PCA fitted on training days only, and joined to each transaction with two surprise scores (amount and timing versus the forecast); contestant C uses only these TSFM columns and contestant D adds them to B's checklist, with B's exact settings, so the only thing that changes is the inputs. Analogy: three chefs get the same kitchen and the same oven; one uses only the old recipe book, one only a new spice blend, one both, and you taste the difference.

No GPU is available, so what runs is set by measured CPU cost: TTM (2 to 3 ms per account-day) on full AMLWorld and the TabFormer sample, Chronos-2 (66 ms or more) on the 20,000-account AMLWorld sample only (methodology M36).

### Model versions pinned (2026-11-14)

Chronos-2 is pinned to Hugging Face commit 29ec3766 and TTM r2 to d6a79570, and the embedding cache refuses to mix outputs from different model versions. The first TSFM validation batch (started at commit 00d2157) loaded TTM as "main", which resolved to the same commit d6a79570 (shown in the run log); its later runs are stamped with the pinning commit but executed the code loaded at start, which differs only in this version metadata.

## Days 18 and 19 (2026-11-14): label-permutation control and explanations

The control retrains the bank-style model on labels shuffled within each day and requires it to score no better than random ranking; if it did better, labels would be leaking into the evaluation some other way. Explanations use LightGBM's built-in SHAP values: each run now lists the columns that drive each model's scores, the share of contestant D's reasoning that comes from the TSFM columns, and the top reasons for the five highest-ranked newly bad accounts. Analogy: a placebo group in a drug trial (the noise-trained model must not "work"), and a doctor who can say which symptoms led to the diagnosis.

## History-length split, report, and agent tools (2026-11-14)

Results are now also split by how much history an account had before the day (fewer than 16 events versus 16 or more, stated in advance in methodology M23), with each day's real review queue kept intact. On the AMLWorld validation day the bank-style model caught 28.1% of the 455 new bad accounts with short histories and 38.7% of the 217 with long histories, so the short-history group is both bigger and harder, and that is where a pretrained model could help most. `tells report <run>` renders every table from the run's files, and the agent's tool layer (Day 21) allows exactly three tools with tests proving that shell access, file paths, unknown configs, extra arguments, and unlocked test scoring are all refused. Analogy: the agent is a lab assistant who can only press three labelled buttons on the machine, and the printout comes straight from the machine.

## Days 22 to 25 (2026-11-14): agent loop, Critic, Reporter, cassettes

The agent is Claude Opus 5.5 in a loop we own: it can only use the three tools, it gets denials back as errors, it stops after 10 tool calls, and it is not allowed to write numbers; it ends with a short machine-readable list of the runs it used and what it claims. The Critic checks those claims against the run files without any model (no typed numbers, canary and control passed, test runs locked, each "better" or "worse" backed by its interval), and the Reporter prints the model's words next to tables generated from the files, marked as a draft for human review. All tests replay a hand-written cassette, so they are free and identical every time; a live run needs an Anthropic API key. Analogy: an intern who drafts the lab report, an automatic checker that rejects any sentence the data does not support, and a supervisor who signs off.

Provenance note: the TSFM validation batch started at commit 6d3ffe1 keeps running the code it loaded then; runs it starts after later commits are stamped with those later commits, whose changes (report, agent, history split) do not touch how those runs compute their numbers.

## First TSFM result, a negative one, and what it revealed (2026-11-14)

On the AMLWorld validation day (run `20261115T031830Z_amlworld_valid_f6ff26be_96919d`), adding TTM features made the bank-style model worse: contestant D caught 19.9% of newly bad accounts on their first bad day at a 0.5% budget against 31.5% for B (difference -11.6 points, 95% interval -15.2 to -8.0), and TTM alone (C) caught 4.8%. The label-permutation control was consistent with chance, so nothing leaked. Splitting by history showed the cause: D matched B on short-history accounts but collapsed on long-history ones (3.2% against 38.7%), because training days hold almost only brand-new accounts (9.3% with 16 or more events) while the validation day is half established accounts (48.4%); a second, fixable cause was that PCA on raw embedding numbers was dominated by a few huge-scale numbers. Analogy: a doctor trained only on newborns is then asked to examine adults, and was also reading the charts with one column in a much larger font.

Changes: standardized PCA, and every run now reports the history mix of training versus evaluation. The first run stays on record and counts as a tried variant.

## TTM with standardized PCA, AMLWorld validation (2026-11-14)

Run `20261115T041547Z_amlworld_valid_f6ff26be_5e9b71`: contestant D (bank-style model plus TTM) caught 37.2% of newly bad accounts on their first bad day at a 0.5% budget against 31.5% for B (difference +5.7 points, 95% interval +2.3 to +9.1, Holm-corrected p 0.006 across three comparisons), and doubled it at the tightest budget (15.0% against 7.9% at 0.1%); the gain appears for both short-history accounts (33.8% against 28.1%) and long-history ones (44.2% against 38.7%). At loose budgets D is slightly behind B (64.0% against 65.2% at 2%, 80.1% against 83.0% at 5%), TTM alone (C) is weak (3.0%), and TTM columns provide 16.5% of D's SHAP attribution, so the TSFM helps as a complement to the checklist, mostly when review slots are scarce. The label-permutation control was consistent with chance.

How to read it: this is the second variant tried on the validation day (the first, unstandardized PCA, made D worse), and the fix was chosen after seeing the first result, so it is a promising validation result, not a finding. Only the sealed test period can confirm it. Analogy: a new detective helps most when the squad can only interview a handful of suspects a day; with time to interview everyone, the old checklist already finds most of them.

## TTM on the TabFormer 300-card sample, validation 2016 (2026-11-15)

Run `20261115T042711Z_small_valid_0f1dcdc6_df305f`: at the 0.1% headline budget, contestant D was 0.8 points behind B (95% interval -2.2 to 0.0, Holm-corrected p 0.31), so no clear difference; A and C were clearly behind. The sample is too small to separate a small effect (29 newly bad cards; about 0.12 review slots per day at 0.1%), so this says "no detectable gain on a small long-history sample", not "no gain". Training and validation both have long histories here (99.4% and 99.9% of account-days with 16 or more events), so the AMLWorld history shift does not apply. The run's elapsed time (61,233 seconds) includes the computer sleeping and is not a compute cost.

## TTM and Chronos-2 on the AMLWorld 20,000-account sample (2026-11-16)

Run `20261115T212743Z_amlworld_small_valid_feecada6_92f7a2`: with only 23 newly bad accounts on the validation day, neither TSFM's contestant D was distinguishable from B (TTM -4.3 points, interval -23.6 to +11.8; Chronos-2 +4.3 points, interval -10.5 to +18.2; Holm-corrected p 1.0 for both), and both TSFM-only contestants were clearly behind. The run mainly shows cost: Chronos-2 took 9,788 seconds of CPU for 78,686 account-days, about 125 ms each, against about 1 ms for TTM. Chronos-2 on the full data remains out of reach without a GPU. Analogy: a microscope that is a hundred times slower may see more, but you cannot tell from a dozen slides.

## Sealed AMLWorld test result (2026-11-16)

Run `20261116T190641Z_amlworld_test_f6ff26be_ac67fd` (test run number 1 under lock `f6ff26be`, sealed in commit 4a21200 before scoring): on the test period (Sept 8 to 10, 1,798 newly bad accounts), contestant D (bank-style model plus TTM) caught 34.5% of newly bad accounts on their first bad day at a 0.5% budget, against 34.4% for B: a difference of +0.1 points, 95% interval -1.6 to +1.9, Holm-corrected p 0.94. The validation gain (+5.7 points) did not hold. Descriptively, D was ahead at the tightest budget (13.6% against 11.0% at 0.1%) and in transaction-level PR-AUC (0.398 against 0.321), and behind at loose budgets (58.8% against 63.1% at 2%, 77.3% against 79.5% at 5%); none of these were pre-registered tests. The canary passed and the label-permutation control scored like random ranking (+0.1 points, -0.2 to +0.4).

What it means: on AMLWorld, adding TTM summaries of account histories did not catch more newly bad accounts at the pre-registered budget. The validation gain came from the second variant tried, chosen after seeing the first, and the sealed test is exactly what protects against that kind of optimism. Analogy: a new training method that looked great on the practice exam, picked after trying two, scores the same as the old one on the real exam, and the real exam is the one that counts.
