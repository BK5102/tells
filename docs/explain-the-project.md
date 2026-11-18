# Explaining Tells: the project step by step

This is the guide to read before talking about Tells with a recruiter, a hiring manager, or an interviewer. It tells the story in the order the work happened. Every step has the same parts:

- **What I did**
- **Why**
- **What else I considered**
- **Cause and effect**: what happened because of this step, including problems it exposed
- **How to say it**: one or two sentences you can say out loud

The technical detail behind each step is in `docs/methodology.md` (numbered M1, M2, ...), the dated diary is in `docs/decisions.md`, and the numbers come from run files under `artifacts/runs/`. This file links them into one story.

## Elevator pitch (final)

Say this in about 30 seconds, then let them pick what to dig into.

- **Question:** "Banks can only review a tiny share of cases each day. I tested whether a pretrained time series foundation model, which summarizes each account's history, helps the tree model a bank already runs catch fraud on the first day an account goes bad."
- **Method:** "I added the foundation model's summary as extra columns to a bank-style LightGBM model and compared it with the same model without them, on a synthetic money-laundering dataset and a synthetic card-fraud dataset, at a fixed daily review budget."
- **Honest result:** "In version 1, IBM's TTM looked like a win on validation but added no detectable gain on two sealed test periods. I diagnosed why, pre-registered two fixes, and tested them on fresh data: one, Amazon's Chronos-Bolt, caught about 3 points more newly bad accounts at the same review budget, statistically significant after correction; the other did not help."
- **Why you can trust it:** "The test plan was hashed and committed before any test score existed, every run plants a fake leak that must be caught, a model trained on shuffled labels serves as a placebo, and the gaps an independent code review found were fixed and merged without touching the sealed runs."

---

## Part 1. The question

### Step 1. Pick a question a bank would actually ask

- **What I did:** framed the project as "does a pretrained time series model (TSFM) add value on top of the model a bank already uses, when analysts can only review a small number of cases each day?"
- **Why:** banks already run tree models on hand-built red flags. A new model only matters if it adds something to that, inside real review capacity. Comparing a TSFM with nothing would prove nothing.
- **What else I considered:** "can a TSFM detect fraud?" Rejected: it ignores the existing system and the review budget.
- **Cause and effect:** this framing decided almost everything later: four contestants (lazy baseline, bank-style model, TSFM only, bank-style model plus TSFM), a daily review budget, and a headline about catches, not accuracy.
- **How to say it:** "I didn't ask whether a foundation model can find fraud. I asked whether it helps the model a bank already has, when the fraud team can only open a fraction of cases each day."

### Step 2. Use the TSFM as a feature supplier, not a replacement

- **What I did:** the TSFM reads each account's history and produces a short numeric summary (an embedding) and a forecast; these become extra columns for the bank's tree model.
- **Why:** this is how published work reports gains (Savvidi et al. 2026), and how a bank would adopt it: add columns, keep the validated model.
- **What else I considered:** fine-tuning the TSFM end to end. Too expensive on a CPU and harder to audit.
- **Cause and effect:** contestant D (bank model plus TSFM) uses exactly contestant B's settings, so any difference comes only from the new columns.
- **How to say it:** "The foundation model is a feature supplier. Same tree model, same settings, extra columns; if the score moves, the columns moved it."

---

## Part 2. Data

### Step 3. Measure whether each dataset can test the idea before using it

- **What I did:** wrote `scripts/check_data_fit.py`, which measures how many transactions each account has before its first fraud, the time span, and the fraud rate. Ran it on five candidate datasets.
- **Why:** a model that reads an account's rhythm needs accounts with a rhythm. Dataset descriptions do not tell you that; measurement does.
- **Cause and effect:**
  - PaySim: 99.9% of accounts appear exactly once. Dropped.
  - IEEE-CIS (real, anonymized): no account id; after rebuilding one, the median account appears once, and 66% of transactions after an account's first fraud are also labeled fraud, a sign labels were copied forward. Dropped.
  - TabFormer: median 1,493 transactions before a card's first fraud. Kept.
  - AMLWorld: median 9 events before laundering, and the last days of the file are about 60% laundering, an artifact. Kept, with the artifact removed.
- **How to say it:** "Before writing a model, I measured whether each dataset could even test the idea. Two popular datasets failed, one because almost every account appears once, one because its labels were copied forward."

### Step 4. Choose two datasets that differ on purpose

- **What I did:** AMLWorld (synthetic money laundering, short histories, 10 usable days) is the main test; TabFormer (synthetic card fraud, decades of history) is the long-history test.
- **Why:** they are opposites on the thing that matters (history length), and AMLWorld is the benchmark a published TSFM paper used.
- **What else I considered:** two card-fraud datasets (TabFormer plus Sparkov). Less variety.
- **Cause and effect:** AMLWorld later turned out to have a training/evaluation history shift (Step 20), which only showed up because the two datasets differ.
- **How to say it:** "I picked a short-history dataset and a long-history one on purpose, so if the foundation model helps on one and not the other, history length is the first suspect."

### Step 5. Load the data into one standard shape, carefully

- **What I did:** one schema for both datasets (account, time, amount, counterparty, category, channel, label), each account's history sorted oldest first. For AMLWorld: each transfer becomes an outgoing event for the sender and an incoming event for the receiver; a self-transfer counts once; rows from Sept 11 on are dropped; 15 currencies are converted to dollars with exchange rates from the first day only.
- **Why:** an investigator looks at an account, and a transfer belongs to both sides. Exchange rates from day one only, because day one is inside every training window, so no later data leaks in.
- **Cause and effect:** the loader reproduces the independently measured counts exactly (24,386,900 TabFormer transactions; 4,522 AMLWorld laundering transfers after truncation), which a test checks.
- **How to say it:** "Even the loader has leak rules: exchange rates come only from the first day, which every model is allowed to see."

---

## Part 3. How results are scored

### Step 6. Score account-days, not transactions

- **What I did:** one case per account per active day, scored by its most suspicious transaction that day.
- **Why:** analysts review accounts, in a daily queue. Scoring transactions would count one burst of ten fraud transactions as ten catches.
- **How to say it:** "The unit is what an analyst opens: an account on a given day."

### Step 7. A daily review budget, with fair tie-breaking

- **What I did:** each day, the top 0.1%, 0.5%, 1%, 2%, or 5% of that day's cases are "reviewed". If cases tie at the cutoff, the remaining slots are shared evenly.
- **Why:** a fraud team has a daily capacity. Sharing ties means a model cannot gain or lose by the order of tied scores.
- **How to say it:** "Fraud is about 1 in 1,000, so accuracy is useless. The question is how much fraud lands in today's review queue."

### Step 8. Headline: caught on the first bad day

- **What I did:** the headline is the share of newly bad accounts caught on their first bad day, at 0.5% of daily cases on AMLWorld and 0.1% on TabFormer.
- **Why:** catching an account the first day it goes bad stops further losses; catching it on day five is worth much less. TabFormer gets the tighter budget because at 0.5% the bank model already catches about 91% there.
- **Cause and effect:** the first version counted accounts already known to be bad (19% of AMLWorld positives, 46% of TabFormer's). Fixed: only genuinely new bad accounts count.
- **How to say it:** "My headline is first-day catch rate, because that is the catch that saves money. And I had to fix it once: it was counting accounts that were already known to be bad."

---

## Part 4. Keeping the exam honest (the Cleanroom)

### Step 9. Build the honesty tools as their own package

- **What I did:** `packages/cleanroom` is a separate Python package with the split, the leak checks, the lock, the metrics, the bootstrap, and provenance. It is released by git tag and shared with two sibling projects (credit risk and payments).
- **Why:** three projects should sit the same exam; copies would drift. A test fails the build if the package ever mentions fraud or imports project code.
- **How to say it:** "The evaluation rules live in a separate package, so all three of my projects are held to exactly the same standard."

### Step 10. Time split with a gap

- **What I did:** train on the past, skip a gap, evaluate on the future. TabFormer: train to Sept 2015, validate on 2016, test April 2017 to 2019, 90-day gaps. AMLWorld: train Sept 1 to 4, validate Sept 6, test Sept 8 to 10, one-day gaps.
- **Why:** a model is used on the future. The gap stands in for the delay before fraud is reported. TabFormer's test spans years because 2017 alone has only 36 fraud cards.
- **How to say it:** "Train on the past, leave a gap for label delay, test on the future."

### Step 11. Prove features only use the past

- **What I did:** 34 red-flag features (activity bursts, unusual amounts, new merchants, money in versus out). A Cleanroom check rebuilds each feature from a copy of the data cut at that transaction's time and requires an exact match.
- **Why:** a feature that peeks at the future makes any model look brilliant.
- **Cause and effect:** I planted a bug (letting windows include their own timestamp) and the check caught it. Same for the TSFM sequences later.
- **How to say it:** "Every feature is tested by replaying history up to that moment and checking nothing changes. I planted bugs to prove the test catches them."

### Step 12. The canary: a planted cheat on every run

- **What I did:** every run adds two cheating features (peeking at the next transaction, peeking at the next label) and refuses to report unless both are caught.
- **Cause and effect:** on its first day it caught its own blind spot: a sample with no laundering accounts made the label cheat undetectable. Samples now always include positive accounts.
- **How to say it:** "You test a smoke detector with a match. Every run holds a match to the leak checks."

### Step 13. Seal the plan before seeing test results

- **What I did:** `tells lock` writes a file named after the hash of the experiment config; it is committed to git before scoring the test period, and test scoring refuses to run without it. Every test run is counted.
- **Why:** pre-registration. Nobody, including me, can change the split or metric after seeing results.
- **How to say it:** "The test plan is hashed and committed before any test score exists. Git history proves the order."

### Step 14. Luck checks

- **What I did:** a paired bootstrap that resamples whole accounts, 1,000 times, and Holm and Benjamini-Hochberg corrections across comparisons.
- **Cause and effect:** the first version re-ranked every resample; duplicated accounts then competed for the same review slots and biased everything downward (one interval did not contain its own estimate). Fixed by forming each day's queue once on real data. Every result now reports the bootstrap mean, so that bias would be visible.
- **How to say it:** "I found a bias in my own confidence intervals, traced it, and fixed it before it could mislead anyone."

### Step 15. Provenance

- **What I did:** every result stores the config hash, git commit, data checksum, library versions, and model versions (pinned to exact Hugging Face commits). Every run is appended to a log, including failures.
- **How to say it:** "Any number in my README can be traced to the exact config, data, model weights, and git commit that produced it, and the record says when a run started with uncommitted changes."

---

## Part 5. Baselines and a ceiling

### Step 16. A lazy baseline and a bank-style baseline

- **What I did:** contestant A ranks by amount (no learning). Contestant B is LightGBM on the 34 features plus category and channel, with fixed settings and no early stopping.
- **Why LightGBM:** gradient-boosted trees are the standard for tabular fraud, handle missing values and categories natively, and train on 16 million rows on a laptop in minutes. No early stopping, because stopping on the validation set would pick the model with the data it is graded on.
- **How to say it:** "The baseline is what a bank would actually run, not a strawman."

### Step 17. The ceiling problem changed the plan

- **What I did:** on full TabFormer, B already caught about 96% of fraud cases at a 1% budget. So I added a 0.1% budget, made first-day catch the headline, and made AMLWorld the main test, where B caught about 31.5%.
- **Cause and effect:** this is why AMLWorld is the main test and why TabFormer's headline budget is 0.1%.
- **How to say it:** "My bank-style baseline was already near perfect on card data, so a gain would have been invisible. I moved the main test to where there was room to improve."

---

## Part 6. The foundation models

### Step 18. Choose the models by measuring cost on my hardware

- **What I did:** checked each model's own code (not its marketing) for an embedding API, then timed them on a laptop CPU.
- **Chronos-2** (Amazon, 119.5M parameters): the model the AMLWorld paper used; official `embed` method; about 66 to 283 ms per account-day on CPU.
- **TTM r2** (IBM, 0.8M parameters): different lab and design; about 1 to 3 ms per account-day.
- **Cause and effect:** with no GPU, Chronos-2 on full data would take days. So TTM runs on full data, and Chronos-2 on a 20,000-account sample. This is stated as a limitation, not hidden.
- **How to say it:** "I chose models by measuring them on my own hardware. The big model was a hundred times slower, so I used the small one on full data and the big one on a sample, and said so."

### Step 19. Turn histories into model inputs

- **What I did:** for each account-day, the account's last 256 events before midnight, as three series: log amount, log time since the previous event, direction (in or out). The model's summary is averaged over the history, standardized, and reduced to 16 numbers with PCA fitted on training days only. A surprise score compares each transaction with the model's forecast.
- **Why:** cutting at midnight is stricter than the checklist's rule, so it cannot leak. PCA on training days only, so later periods cannot shape the inputs.

### Step 20. The first TSFM result was negative, and why

- **What happened:** the first full AMLWorld run with TTM made the bank model worse: D caught 19.9% of new laundering accounts on day one against B's 31.5% (-11.6 points).
- **How I diagnosed it:** split the result by history length. D matched B on short-history accounts but collapsed on long-history ones (3.2% against 38.7%).
- **Two causes found:**
  1. AMLWorld starts on Sept 1, so training days hold almost only new accounts (9.3% with 16 or more earlier events), while the validation day is half established accounts (48.4%). The model learned from newborns and was graded on adults.
  2. My PCA ran on raw embedding numbers, where a few numbers varied hundreds of times more than the rest, so one direction took 97% of the variance.
- **What I did:** standardized before PCA; every run now reports the training versus evaluation history mix; the first run stays on record and counts as a tried variant.
- **How to say it:** "My first result said the foundation model made things worse. Instead of dropping it, I split it by history length, found a train/test shift built into the dataset and a scaling mistake of mine, fixed the mistake, kept the failed run on record, and sealed the plan before touching the test data."

### Step 21. Controls and explanations

- **What I did:** a label-permutation control (the bank model retrained on shuffled labels must score like random ranking) and SHAP explanations (which inputs drive each score, and how much of D's reasoning comes from the TSFM).
- **Cause and effect:** every AMLWorld control, and the TabFormer test-period control, scored like random ranking, so no hidden leak there. The one exception, TabFormer validation, is explained in Step 24. On full AMLWorld, TTM columns provide 16.5% of D's SHAP attribution.
- **How to say it:** "A placebo group for the model, and an explanation for every flag."

---

## Part 7. Results

### Step 22. Validation results (before sealing)

| Headline: first-fraud-day catch | Full AMLWorld, 0.5% | TabFormer sample, 0.1% | AMLWorld sample, 0.5% |
|---|---|---|---|
| B. Bank-style model | 31.5% | 9.5% | 39.1% |
| D. B + TTM | 37.2% (+5.7 [+2.3, +9.1]) | 8.7% (no clear difference) | 34.8% (no clear difference) |
| D. B + Chronos-2 | not run (needs a GPU) | not run | 43.5% (no clear difference) |

- **Read it as:** on full AMLWorld, the TSFM helped, mostly when review slots are scarce (15.0% against 7.9% at 0.1%), and slightly hurt at loose budgets (80.1% against 83.0% at 5%). TTM alone was weak everywhere. The samples were too small to see an effect of this size.
- **Caveat to say out loud:** this was the second variant tried, chosen after seeing the first. That is why the test period was sealed next.

### Step 23. Sealed test results (AMLWorld and TabFormer)

- **What I did:** sealed each dataset's plan (split, headline, contestants, TSFM settings in the config) by hash; settings that live in code, such as standardizing before PCA, are pinned by the git commit stamped on the run. I committed the lock files, then scored each test period exactly once. The TabFormer plan was sealed at the same time as AMLWorld's, before any full-data TSFM run on it.

| Sealed test, headline: first-bad-day catch | AMLWorld (Sept 8 to 10), 0.5% budget | TabFormer (April 2017 to 2019), 0.1% budget |
|---|---|---|
| Newly bad accounts | 1,798 | 243 cards |
| B. Bank-style model | 34.4% | 11.2% |
| D. B + TTM | 34.5% | 13.2% |
| D minus B, 95% interval | +0.1 points [-1.6, +1.9] | +2.0 points [-0.6, +5.0] |
| Holm-corrected p | 0.94 | about 0.16 |
| Verdict | no clear difference | no clear difference |
| Lock, test run number | `f6ff26be`, run 1 | `701e5f22`, run 1 |

- **Descriptive details (not pre-registered):** on AMLWorld, D was ahead at the tightest budget (13.6% against 11.0% at 0.1%) and on transaction-level ranking (PR-AUC 0.398 against 0.321), and behind at loose budgets (58.8% against 63.1% at 2%). On TabFormer, D was ahead at every budget (for example 33.9% against 29.9% at 0.5%, 77.0% against 72.4% at 5%), but none of these were the tested comparison.
- **Combined message:** across both sealed tests, TTM added no detectable gain at the pre-registered budgets. TabFormer leans in D's favor, but its interval includes zero, and "leans" is not a finding.
- **Cause and effect:** the AMLWorld validation gain (+5.7 points) came from a variant chosen after seeing the first validation run, on a single day. The sealed test, with 2.7 times as many new bad accounts and no choices made on it, did not confirm it. This is the protection the lock was built for, and it worked.
- **Integrity checks:** both runs passed the canary, and in both the bank model trained on shuffled labels scored like random ranking (TabFormer: 0.5% against 0.1% for random, interval for the difference -0.1 to +1.4 points). The TabFormer test run started from a clean, committed checkout.
- **How to say it:** "I sealed two test plans before seeing any test data and scored each one once. On both, adding the foundation model gave no detectable gain at the budget I had committed to: plus 0.1 points on money laundering and plus 2 points on card fraud, with intervals that include zero. My validation result had looked like a win, and the sealed tests are exactly what stopped me from reporting it as one."

Where to point people: `docs/findings.md` (the full write-up), `docs/analysis/amlworld-test-descriptive.md` (every budget with intervals, each day, who catches whom; generated by `scripts/analyze_test_run.py`, not produced yet), and `docs/audits/2026-11-16-sealed-test-audit.md` (proof the test was sealed first and scored once).

### Step 24. A control that failed on TabFormer validation, and what I did about it

- **What happened:** on TabFormer validation (2016, run `20261116T193405Z_tabformer_valid_701e5f22_d9a420`), D was +2.5 points over B (31.0% against 28.4%, interval -1.1 to +6.0, 556 newly bad cards): no clear difference. But the label-permutation control failed: the bank model trained on labels shuffled within each day caught 5.9% on the first bad day, against 0.1% for random ranking (difference +5.8 points, interval +4.0 to +7.8). A placebo is not supposed to work.
- **Why this can happen without a leak:** the control shuffles labels only inside each day. That keeps how many fraud labels each day has, so it still tells the model which days were fraud-heavy. TabFormer fraud comes in bursts on particular days, so a model can learn "transactions that look like a burst day are riskier" without ever seeing whose transaction was fraud. Analogy: shuffle the names on each day's list of failed students but keep how many failed each day; if failures cluster on exam days, you can still learn "exam days are bad" without knowing anyone's grade.
- **Why I did not just assume that:** the other leak defenses passed on this run (canary caught both planted cheats; features are proven past-only by replay), and the same control on the TabFormer test period scored like random ranking. That points to day-level timing, not a leak, but "points to" is not proof.
- **What I did:** added a stricter control that shuffles labels across all training days, which also breaks the link between labels and days. If that one scores like random, the within-day excess came from fraud bursts, not a leak. It was run after the sealed tests and does not change them: the sealed results are what they are, and this is a diagnostic next to them.
- **What else I considered:** dropping the within-day control (it hides the problem, and dropping a check after it fails is the opposite of honest); treating the failure as proof of a leak and discarding the run (not supported by the other checks).
- **Outcome of the stricter control:** it shrank the problem a lot but did not remove it. Rerun side by side (`docs/analysis/tabformer-shuffle-scope-control.md`), the within-day placebo beat random by +5.8 points at 0.1% (interval +4.0 to +7.7); the across-all-days placebo by only +1.5 (+0.6 to +2.6) and, with a second random seed, +0.9 (+0.3 to +1.6). So day-level fraud bursts explained most of the original excess, as I suspected.
- **Why the small remainder is still not a leak:** with labels shuffled across all days, the placebo has no label information at all, so no label can sneak through training. Whatever edge it has must come from the features alone. I found two label-free reasons, measured from the run's saved cases:
  1. **Busy days score higher.** A case's score is the highest score among that day's transactions, so an account-day with more transactions gets more lottery tickets, even if every score is random. Newly bad account-days average 4.0 transactions against 2.0 for all account-days. Ranking only by transaction count catches 0.2% at 0.1% (random: 0.1%) and 1.8% at 0.5% (random: 0.5%). This explains a small part.
  2. **A noise-trained tree still flags oddballs.** Even fitted to random labels, a tree model gives its most extreme scores to rare, unusual-looking transactions, and TabFormer fraud looks unusual, so the placebo works like a crude anomaly detector. Analogy: a guard dog that was never shown a burglar still barks at strangers, and burglars are strangers. This explains the rest; it fits the numbers but I have not proven it with its own experiment.
  The leak checks agree: the features passed the past-only replay proofs and the canary on the same run, the sealed TabFormer test's placebo was consistent with chance (+0.4, -0.1 to +1.4), and on AMLWorld every control passed.
- **Cause and effect:** the lesson is about the yardstick, not about a leak. "Beat a constant score" is too easy a bar for "chance" when fraud is bursty and looks unusual. A stronger control would compare the real model with the spread of many label-free placebo models, so "chance" becomes a range instead of one number. The v0.1 sealed results stand unchanged; this is a diagnostic next to them (methodology M42, limitation L24).
- **How to say it:** "My placebo control failed on one dataset, and instead of explaining it away I ran a stricter placebo with no label information at all. It still beat random a little, and I traced that to two label-free effects, busy days getting more chances and a noise-trained tree still flagging unusual transactions, so the honest conclusion was not a leak but that 'beats random' was too weak a bar, and I said so."

### Step 25. The baseline goes stale over time

- **What I saw:** B caught 90.1% of newly bad cards on their first bad day at a 0.5% budget on TabFormer validation (2016), but 29.9% on the test period (April 2017 to 2019). At the 0.1% headline budget it fell from 28.4% to 11.2%.
- **Why (most likely):** the model is trained on data through September 2015 and never updated. The test period starts 18 months later and runs for almost three years, so the fraud patterns it learned have drifted further and further from what it is scored on. The test window is also a different mix of fraud cases, so this is the likely main cause, not a measured one.
- **Cause and effect:** this is why banks retrain models on a schedule. It also changes how to read the TabFormer test: both B and D were judged as stale models, which is a fair comparison between them but not how a bank would deploy either.
- **What else I considered:** retraining during the test period. Not allowed: the sealed plan fixes training to end in 2015, and changing it after seeing results would break the seal. A rolling retrain (retrain every few months, test on the next few) is the natural next step, under a new sealed plan.
- **How to say it:** "My baseline caught 90% on the year right after training and 30% on the years after that, because a model trained through 2015 goes stale. That is why banks retrain, and a rolling retrain is my next experiment."


---

### Step 25b. Why the foundation model did not help: the diagnostics

- **What I did:** after both sealed tests, I ran six descriptive checks on the existing runs to explain the result, not to look for a win (`docs/analysis/why-no-gain.md`, generated by `scripts/diagnose_no_gain.py`).
- **What they showed, most important first:**
  1. **Same-day blindness.** The bank-style model gets 79% (AMLWorld) and 70% (TabFormer) of its SHAP attribution from the transaction itself and the last 24 hours. The TSFM only reads history up to midnight, and for 52% (AMLWorld) and 64% (TabFormer) of newly bad accounts the first fraud is the day's first transaction. The signal that catches first-day fraud happens today, where the history summary cannot look.
  2. **Little history on AMLWorld.** 51% of newly bad accounts in the AMLWorld test had fewer than 16 earlier events, so the summary had little to read.
  3. **Weak signal on its own.** TTM alone ranks transactions only about 3.4 times better than chance (PR-AUC against the base rate), against 100 to 289 times for the bank-style model.
  4. **Partly redundant.** The checklist already explains 42% (AMLWorld) and 28% (TabFormer) of the TSFM summary in a linear fit, a lower bound.
  5. **Reshuffles rather than adds.** On the AMLWorld test, 140 newly bad accounts were caught only with the TSFM and 138 only without it; on TabFormer, 10 and 5.
  6. **Power.** The AMLWorld test could detect a gain of about 2.5 points, so its +0.1 really is near zero. The TabFormer test needed about 4 points, so its +2.0 means "small or none, too few cases to tell", not "zero".
- **What else I considered:** trying more models on the spent test sets (that would be fishing; v0.2 does it properly on fresh data).
- **Cause and effect:** the diagnosis shaped v0.2's ideas and their limits: recent pooling attacks the averaging, but no history summary can see today's transactions, so a large gain was never likely. Domain mismatch (models pretrained on smooth series, applied to irregular bursts) is listed as a hypothesis only.
- **How to say it:** "The model that catches first-day fraud mostly looks at today's transactions, and a foundation model reading yesterday's history cannot see today. I measured that: about three quarters of the bank model's signal comes from the last 24 hours, and most first frauds are the day's first transaction."

---

## Part 8. The agent, and a final honesty check before release

### Step 26. An AI assistant that runs experiments but cannot cheat

- **What I did:** Claude Opus 5.5 in a loop I own, with three tools only (run an experiment by config name, read a run's summary, get chart data), a 10-call budget, and a rule that it may not write numbers. A Critic checks every claim against the run files without any model; a Reporter puts the model's words next to tables generated by code, marked as a draft for human review.
- **Why a loop I own:** every call must pass my allow-list and every exchange must be recordable for free, repeatable tests.
- **Why no numbers from the model:** language models can invent numbers; numbers come only from files.
- **How to say it:** "The agent can plan and run experiments (tested offline with a recorded conversation so far), but it cannot read raw data, cannot open the sealed test without the lock, and cannot type a number. A checker rejects any claim the data does not support."

### Step 27. An independent code review and docs audit before release

This is the last layer of the honesty machinery from Part 4: checking the checkers.

- **What I did:** while the sealed runs were going, two AI agents worked in parallel: one reviewed the code for gaps in the guardrails, one audited the docs for statements that were stale or claimed too much.
- **What the code review found (7 issues):**
  1. The TSFM cache could return duplicated rows if the chunk size changed between runs.
  2. The report command could write an incomplete feature cache.
  3. Test scoring did not require the code to be committed.
  4. The agent could ask for the test period on its own.
  5. The Critic could be fooled by numbers spelled out in words, and could misjudge "better" or "worse" because it did not use the Holm correction.
  6. The canary used test-window labels to choose which accounts to sample.
  7. One tool error could crash the agent's whole loop.
- **What the docs audit found:** stale statements and overclaims. The main one: docs said the lock seals the code, but it hashes the config file only. Code settings, such as standardizing before PCA, are pinned by the git commit stamped on each run, and the docs now say so.
- **How the fixes went in:** each fix got a regression test (a test that fails on the old code and passes on the fixed code), on a separate branch in its own git worktree. The branch was merged only after the sealed runs finished.
- **Why hold fixes on a branch:** a run stamps the git commit it started from. If I had changed the main checkout mid-run, the stamp would point to code the run never used, and the provenance record would quietly lie. Keeping fixes on a branch kept every sealed run's stamp true.
- **Cause and effect:** none of the seven issues changed a reported number; they closed doors nobody had walked through yet. Details: `docs/decisions.md`, 2026-11-16.
- **What else I considered:** fixing on the main checkout right away (breaks provenance of running jobs); waiting to review until after release (ships known gaps).
- **How to say it:** "Before release I had my own guardrails independently reviewed. It found seven gaps, I fixed each with a test that proves it, and I kept the fixes on a separate branch until the sealed runs finished so their records would point to the exact code they ran."

---

## Part 8b. Version 2: a pre-registered follow-up

The full plan is in `docs/v2-plan.md`. It was written and fixed on 2026-11-17, before any v0.2 model ran on any data, and amended the same day, still before any result, when MOMENT proved too slow (Step 32). The two variants are `ttm_recent` and `chronos_bolt`. Status: in progress; no v0.2 result exists yet.

### Step 28. Step back after a negative result, instead of trying more models on the same test

- **What I did:** stopped and wrote a new pre-registered plan (v0.2) rather than running new variants on the two test periods v0.1 had already used.
- **Why:** both v0.1 test periods are spent. If I tried new embeddings on them until one beat B, one would win by luck sooner or later, and it would look like a finding without being one. That is p-hacking. But the diagnosis of v0.1 suggested specific reasons the signal might have been watered down (Step 29), and those deserve a real test. The honest way to ask a follow-up question is a new plan, fixed before any result, on data nobody has touched.
- **What else I considered:** stopping at v0.1 (honest, but leaves cheap, specific explanations untested); trying variants on the old test periods (fishing); promoting v0.1's descriptive hints to findings (they were never pre-registered).
- **Cause and effect:** v0.1's results stay exactly as they are. v0.2 changes no Cleanroom code, so the sibling projects are unaffected, and the new TSFM adapter follows the existing interface so it can be reused. Keeping "a sealed test is scored once" true protects all three projects, which make the same promise.
- **How to say it:** "My first answer was an honest negative. Instead of trying new models on the test I had already used, which would be fishing, I wrote a new sealed plan to test the two most likely reasons, on data no model had seen."

### Step 29. Two ideas, chosen by reasoning, not by results

- **What I did:** fixed exactly two variants, each with its own C (TSFM only) and D (B plus TSFM), all compared with B under a Holm correction.
  1. **`ttm_recent`:** the same TTM, but the summary comes from the most recent patch of the history instead of the average over all patches (a patch is a fixed-size chunk, like one page of a bank statement), and 64 standardized PCA components instead of 16.
  2. **`chronos_bolt`** (first planned as `moment`, swapped before any result, see Step 32): Chronos-Bolt-tiny, a small open model from Amazon (Apache-2.0) in the same family as the Chronos-2 model the AMLWorld paper used, with a different design from TTM. Same downstream setup: 64 standardized PCA components into the same LightGBM, plus surprise scores from its forecast. The original choice, MOMENT-1, was a model built to summarize whole series rather than forecast; it turned out too slow for this laptop.
- **Why:** fraud is a change in behavior, and averaging a whole history blurs a recent change. TTM reads a 512-step window padded on the left, and AMLWorld accounts have about 10 real events, so most patches were padding and the average watered down the real signal. 16 components may also discard rare patterns. Separately, v0.1's answer might be about TTM alone; a different pretrained model checks that. The two variants test two different explanations: "wrong summary" and "wrong model". (The first plan asked "wrong kind of model?" with MOMENT, a summarizer instead of a forecaster; Step 32 explains why that moved to future work.)
- **What else I considered, and why not now:**
  - Chronos-2 on full data: no GPU; on CPU it would take days.
  - Graph features: they ask a different question (does the money network help?), not whether account histories add to a bank model.
  - A supervised sequence model trained end to end on fraud labels: a different approach with many more settings to choose, slow on CPU; kept for a later plan.
  - More pooling choices (max, last few patches, average of real patches only): each extra variant adds a comparison and weakens the correction. Two reasoned variants are a sharper test than many guessed ones.
- **Cause and effect:** if `ttm_recent` helps, dilution was the problem. If only `chronos_bolt` helps, the choice of pretrained model mattered more. If neither helps, the case against small frozen TSFM summaries for this task is stronger. (With MOMENT the second question was "summarizer instead of forecaster?"; that question moved to future work with the swap.)
- **How to say it:** "I picked two changes from the diagnosis, not from the scores: stop averaging a history that is mostly padding, and try a different pretrained model, from the family the published paper used."

### Step 29b. Building the two v0.2 options in code

- **What I did:** built two new model options without changing the v0.1 code. `ttm_recent` uses TTM's summary of the most recent 64 events instead of the average over its whole 512-step window. `moment` adds MOMENT-1-small, a pretrained model that learned by filling in hidden parts of time series (like a fill-in-the-blanks exercise) instead of forecasting.
- **Why:** fraud is a change in behavior, and averaging blurs a change into the past. Also, most AMLWorld histories are about 10 events in a 512-step window, so the v0.1 average was mostly an average of padding. A third model from a different lab checks that the v0.1 answer is not about TTM alone.
- **Cause and effect:** the new options get their own names, so their features and caches can never mix with v0.1's. MOMENT gives no forecast, so it has no surprise score; it is about 60 times slower than TTM on CPU (143 ms against 2.5 ms per account-day), which limits it to the AMLWorld sample. Both are untested on real runs yet. Update: that cost later ruled MOMENT out of v0.2, and Chronos-Bolt-tiny took its place before any result (Step 32).
- **What else I considered:** larger MOMENT sizes (too slow on CPU), weighting recent patches (one more knob to tune), fine-tuning (no longer a frozen model). Details: methodology M45 and M46.
- **How to say it:** "After a clean negative result, I wrote down two specific reasons it might have missed, the averaging and the choice of model, and built a test for each without touching the code behind the published numbers."

### Step 30. Fresh data for the test, old data for development

- **What I did:** the sealed test uses AMLWorld LI-Small, a sister dataset from the same generator with a lower laundering rate, which no model of mine has touched. Same calendar split as v0.1 (train Sept 1 to 4, validate Sept 6, test Sept 8 to 10, one-day gaps), if the data fit check confirms it works. All debugging happens on HI-Small, which v0.1 already used, on its 20,000-account sample (see Step 32 for why not the full day). The fit check confirmed the plan: LI-Small has the same end-of-file artifact on the same date, so it is cut at 2022-09-11, and the v0.1 calendar leaves 580 newly bad account-days in validation and 1,549 in test. The v0.2 config is sealed with `tells lock` and committed before any model runs on LI-Small; then LI-Small validation and test are scored, the test once.
- **Why fresh data:** a follow-up question needs an exam the student has not seen. Why LI-Small: it asks the same question under the same rules, and there is no untouched TabFormer period left (2016 and 2017 to 2019 are used, and 2020 has no fraud). A lower laundering rate is also closer to a real bank.
- **Why development stays on HI-Small:** code has to be checked somewhere. A day that is already spent cannot be spent again, so all checking happens there and LI-Small arrives untouched at sealing time. No setting is tuned on LI-Small, not even on its validation day.
- **Why the headline stays at 0.5%:** v0.1 hinted at a gain at 0.1%, but moving the headline to where a hint appeared is choosing the metric after seeing data. 0.1% is reported next to it as descriptive.
- **What else I considered:** a held-out slice of HI-Small (every usable day is already used); larger AMLWorld files (several times more account-days to embed on a CPU).
- **Cause and effect:** details that depended on compute or on the data check (the second model and its revision, the LI-Small truncation date, the config path `configs/amlworld_li_v2.yaml`) were left open in the plan and filled in on 2026-11-17, before sealing and before any result; each change is listed with its reason in the plan's "Changes before sealing" section.
- **How to say it:** "I debugged on data I had already used and sealed the plan before the new data was ever touched, with the same headline as before, so nothing in the test could be chosen by looking at it."

### Step 31. What would and would not count as a win

- **A win:** at least one D variant beats B on the LI-Small sealed test, first-fraud-day catch at 0.5%, with a Holm-adjusted p below 0.05 and an interval above zero, on a run whose canary and placebo pass.
- **Not a win:** D ahead with an interval that includes zero; ahead only at other budgets; ahead on validation but not test; any HI-Small gain.
- **A second negative result** is a legitimate outcome and is reported exactly like a win: same tables, same places, v0.1 left untouched.
- **Why decide this now:** if the definition of success is written after the result, any result can be called a success.
- **How to say it:** "Before running anything, I wrote down what would count as a win, and I committed to reporting a second negative exactly the same way."

### Step 32. Dropping MOMENT, and how the replacement was chosen

- **What I did:** measured MOMENT-1-small before committing compute to it: 143 ms per account-day on this laptop's CPU, the same for every history length because it always reads a 512-step window. v0.2 needs about 2.0 million HI-Small and 2.8 million LI-Small account-days, so about 190 CPU hours, and I have no GPU. I dropped it from v0.2, kept it as future work with a GPU, and replaced it with Chronos-Bolt-tiny (Amazon, Apache-2.0, about 9M parameters, revision `a0e552de83495b5c28c14c71c374f3e33280b340`). All of this happened before any v0.2 result existed.
- **Why measure first:** a plan that assumes a model will run is a promise I might not keep. Timing a small batch takes minutes; finding out after a week of computing would not. Analogy: checking whether the furniture fits through the door before carrying it up the stairs.
- **Why a stated rule for the replacement:** once a slot opens up, picking a model by feel leaves room to pick by hope. So I wrote the rule first: the cheapest TSFM from a different family than TTM that gives embeddings and runs within 10 ms per account-day on this CPU. Then I measured. TTM's short-context versions need a "frequency" input (like "hourly" or "daily") that irregular transaction histories do not have, so they could not be used. Chronos-Bolt-tiny took about 1.2 ms per channel, about 3.6 ms for three channels, and qualified.
- **Why not run MOMENT on a sample instead:** a 20,000-account LI-Small sample would leave only about 40 to 50 newly bad accounts in the test. My full sealed tests, with 1,798 and 243 newly bad accounts, could only tell apart differences of about 2 to 3 points; a test that small could not detect a gain of a few points, so it would say "no clear difference" whatever the truth. A test that cannot come out positive is not worth sealing.
- **Why Chronos-Bolt-tiny is a sensible stand-in:** it is from the Chronos family, whose large model (Chronos-2) the AMLWorld paper used and which I could not run on full data; and it is built differently from TTM, so it still tests whether v0.1's answer was about TTM alone. It reads one series at a time, so each of the three channels gets its own summary and they are joined (768 numbers), and its median next-step forecast gives the surprise score.
- **Also changed before sealing:** development moved from the full HI-Small validation day to the 20,000-account HI-Small sample. The plan forbids tuning on HI-Small, so development only has to prove the code runs end to end; the full day would cost hours for numbers I am not allowed to use.
- **What else I considered:** Chronos-Bolt small (slower than needed; the rule asks for the cheapest that qualifies); keeping only one variant (leaves "is it TTM-specific?" untested when a cheap model qualifies).
- **Cause and effect:** v0.2 no longer asks "summarizer versus forecaster?", because Chronos-Bolt forecasts; that question waits for a GPU. The number of variants and comparisons, and so the correction, stay the same. MOMENT never ran on any data, so it is not a tried variant and dropping it was not a choice made from results. Details: `docs/v2-plan.md` ("Changes before sealing"), methodology M46 to M48.
- **How to say it:** "I timed my planned model before trusting the plan, found it needed 190 CPU hours, and rather than shrink the test until it could not detect anything, I wrote a rule for the replacement before seeing any result and let the rule pick the model."

---

### Step 33. The v0.2 result: a modest, pre-registered win for one variant

- **What I did:** scored the sealed LI-Small test once, after committing the lock (`b8a1927`). Run `20261117T235515Z_amlworld_li_v2_test_cf534a03_0f604f`.
- **What happened:** B plus Chronos-Bolt-tiny caught 24.0% of the 1,549 newly bad accounts on their first bad day at the 0.5% budget, against 20.7% for B: +3.2 points, 95% interval +1.1 to +5.4, Holm p 0.016. B plus TTM recent caught 21.6%: +0.8, interval -1.2 to +2.8, no detectable gain. The canary passed and the shuffled-label control scored like random. This meets the win rule I wrote in Step 31.
- **What it does and does not mean:**
  - **It does mean:** on fresh data that no choice touched, one pretrained model added real signal to the bank-style model at a tight review budget.
  - **It does not mean** that "TSFMs help". The reworked TTM did not.
  - **It does not show real-bank performance.** The data is synthetic.
  - **It does not say why** Chronos-Bolt helps. It forecasts with quantiles and summarizes each channel separately, but neither was tested as the reason.
- **The part to be upfront about:**
  - **Validation pointed the other way.** On the validation day it was -2.1 points, interval -5.9 to +1.4. That day has 580 accounts against 1,549 in the test, and the plan named the test as the number that counts before either was scored.
  - **Behind at loose budgets.** D was behind at 2% and 5% (descriptive), the same trade-off as v0.1.
  - **Four sealed tests in all.** Correcting for every "B + TSFM" test the project ever sealed, four in total, the result still holds (Bonferroni 0.032).
- **Cause and effect:** v0.1's negative was partly about the model, not the idea. That is exactly what the second variant was there to check.
- **How to say it:** "My first model gave a clean negative. I diagnosed it, wrote down two fixes and a win rule before touching new data, and one fix, a different pretrained model, gave a 3-point gain that survives correction. I'm careful to say it's one model on one synthetic dataset, and that the validation day pointed the other way."

### Step 34. Cleanroom v0.2: new exam tools for the sibling projects

- **What I did:** added version 0.2.0 of the shared Cleanroom package with tools that Reserve and Culprit need: scores for forecasts that give ranges instead of one number, a test of whether one forecaster really beats another over time, backtests from many start dates, lookups that use each figure as it was first published, alarm scoring (how fast faults are caught and how many false alarms), a guard for sealed runs, and a vault that keeps the answer key shut until a sealed run has finished. Tells' own code was not touched.
- **Why:** the honesty rules should be written once and shared, so a sibling cannot get them subtly wrong in its own copy (methodology M27, M50).
- **What else I considered:** letting each sibling write its own versions; using Newey-West weights in the Diebold-Mariano test, which a simulation showed rejects a true "no difference" about 9% of the time instead of 5%, so equal weights are the default.
- **Cause and effect:** the siblings can now be graded by the same exam board as Tells, including on forecasts and alarms that Tells never needed.
- **How to say it:** "I extended the shared exam toolkit for my two sibling projects, and a simulation in its tests caught that the textbook variance choice would have made one of the new tests too eager to declare a winner."

---

## Part 9. Every tool and framework, and why

| Tool | What it does here | Why this one |
|---|---|---|
| Python 3.11+ | Everything | Standard for data science and ML |
| uv | Installs packages, locks versions, runs commands | Fast, one lockfile for reproducible installs, supports a workspace with the Cleanroom as its own package |
| Polars | Data frames | Much faster than pandas on 24 million rows, and its lazy reading keeps memory down |
| NumPy | Arrays, bootstrap, PCA | Standard numeric core |
| LightGBM | Contestants B, C, D | Standard for tabular fraud; fast on CPU; native categories and missing values; built-in SHAP |
| PyTorch (CPU build) | Runs the TSFMs | Required by both TSFM packages; CPU wheels are smaller and no GPU is available |
| chronos-forecasting | Chronos-2 | Amazon's official package, with an official embedding API |
| granite-tsfm | TTM r2 | IBM's official package for TTM |
| pytest | 187 tests (more after the review fixes) | Standard; supports skipping data-dependent tests in CI |
| ruff | Lint and formatting | One fast tool for both |
| GitHub Actions | CI on Python 3.11 and 3.12 | Free for the repo; every push is tested |
| Anthropic Python SDK | The agent's model calls | Official SDK for Claude; optional install so the rest needs no key |
| Hugging Face Hub | Model downloads | Where both TSFMs are published; pinned to exact commits |
| git worktrees and branches | Kept the review fixes in a separate folder and branch while sealed runs were going | A run stamps the commit it started from; editing the main checkout mid-run would make that stamp point to code the run never used (Step 27) |
| Parallel AI agents | One reviewed the code, one audited the docs, at the same time | An independent reader catches what the author misses, like a second proofreader; running them in parallel fit the review into the time the sealed runs took |

---

## Part 10. Hard questions and honest answers

**"So did the foundation model help or not?"** Version 1, no: IBM's TTM gave no detectable gain on two sealed tests. Version 2, yes for one model: on a fresh sealed test, Amazon's Chronos-Bolt added 3.2 points (interval +1.1 to +5.4, Holm p 0.016), while the reworked TTM gave nothing detectable. One-sentence version: "One pretrained model helped modestly on fresh data under rules written in advance; the other did not, so the answer depends on the model."

**"Your v0.2 validation day said the opposite. Why believe the test?"** Because the plan said in advance that the test decides, and the test is bigger: three days and 1,549 newly bad accounts, against one day and 580. I report both. If I had used validation to choose, I would have dropped Chronos-Bolt and missed the result the larger sample supports. That said, the disagreement is a reason to replicate before relying on it.

**"If an expert reviewed this, what would they say you missed?"** The caveats I list with the result (one synthetic dataset, a modest effect, validation pointing the other way, model-specific, behind at loose budgets) are already disclosed, which is the expected standard. What an expert would still ask, and I have not tested: whether the gain comes from pretraining or just from more history columns (compare with simple history statistics and an untrained copy of the network); how much training-seed and day-to-day variation would widen the interval; and whether a second generator or real data would show the same. All of it is in `docs/expert-review.md`. One-sentence version: "I wrote my own expert review: the disclosed caveats are handled, and the open questions are whether pretraining is the cause and how much extra uncertainty training randomness adds, which are my next experiments."

**"Isn't this all synthetic data?"** Yes, from two IBM generators. Real transaction data is private. I measured each dataset before using it, removed an artifact, and say everywhere that results do not show real-bank performance.

**"Your validation gain came after you changed something. Isn't that p-hacking?"** It was the second variant tried, chosen after the first; I recorded both, counted them, and sealed the plan before scoring the test period, which is the number that counts.

**"Why not a graph model? The best AMLWorld papers use the transaction network."** Out of scope: the question is whether account histories add to a bank-style model. My baseline has money-movement red flags so it is fair, but it is weaker than the best graph models, and I say so.

**"Why didn't you run the big model on everything?"** No GPU. I measured it at about 125 ms per account-day on CPU; full data would take days. I ran it on a sample and labeled that.

**"So the project failed?"** No. The question was whether a TSFM adds value at a fixed review budget, under an evaluation that cannot be gamed. v0.2 then found a modest gain for Chronos-Bolt on fresh data. The answer, for TTM on both v0.1 sealed tests, is "not detectably": +0.1 points on AMLWorld (interval -1.6 to +1.9) and +2.0 points on TabFormer (interval -0.6 to +5.0). A clean negative result from a rigorous setup is more useful to a fraud team than an unverified positive one, and the hints (D ahead at the tightest AMLWorld budget, ahead at every TabFormer budget) say where a new sealed test should look.

**"How do you know there is no leakage?"** Features are proven past-only by replay; every run plants two cheats that must be caught; a model trained on shuffled labels must score like random, and did on every run except TabFormer validation (next question); the test plan is sealed by hash before scoring.

**"Your control failed on TabFormer validation. Doesn't that mean a leak?"** Not necessarily, and I did not assume either way. The control shuffled labels within each day, which keeps how many frauds each day had, and TabFormer fraud comes in bursts on particular days, so a model can learn "burst days are risky" without any leak. The canary and the replay checks passed on that run, and the same control passed on the TabFormer test period. To test the explanation, I ran a stricter control that shuffles labels across all days. Outcome: the excess at 0.1% fell from +5.8 points to +1.5 and +0.9 (two seeds), so day-level bursts were most of it. The rest cannot be a leak, because a model trained on labels shuffled across all days holds no label information; its edge comes from the features. Two label-free effects explain it: a case takes the highest score of its day, so busy account-days (newly bad ones average 4.0 transactions against 2.0) get more chances even under noise, and a tree fitted to noise still gives extreme scores to unusual transactions, which TabFormer fraud tends to be (the second effect fits the numbers but is not separately proven). The conclusion is that a constant score is too weak a stand-in for chance on this kind of data; a stronger control would compare against many label-free models. One-sentence version: "My control failed, I found out why, and it was not a leak: the bar for 'random' was too low, and I documented a stronger one."

**"Why did your baseline drop from 90% to 30%?"** On TabFormer at a 0.5% budget, B caught 90.1% in 2016 and 29.9% in April 2017 to 2019. It was trained through September 2015 and never updated, so it went stale as fraud patterns drifted; the test also covers a different set of fraud cases, so staleness is the likely main cause, not a measured one. That is why banks retrain, and a rolling retrain under a new sealed plan is my next step.

**"How do you know your own code did not have bugs?"** Four layers. 187 automated tests, including tests against hand-worked answers. Planted bugs, to prove the leak checks catch them (Steps 11 and 12). An independent code review before release, which found seven gaps (Step 27). And a regression test for every fix, which failed on the old code before it passed on the new. That does not prove zero bugs; it means each kind of bug I could think of has a test that would catch it.

**"What went wrong?"** My first TSFM result was negative because of a dataset shift and a scaling mistake; my first confidence intervals were biased; my first headline counted known-bad accounts; my first canary had a blind spot. Each was found by a check, fixed, tested, and documented.

**"What would you do next?"** Run Chronos-2 on full data with a GPU, test on real data under an agreement, and try a rolling retrain so the model stays fresh (Step 25) and sees established accounts during training.

**"Could your safeguards still be bypassed?"** A code review on 2026-11-16 found seven gaps, none of which changed a reported number: for example, an uncommitted lock was accepted, the agent could choose to spend the test run, and the Critic judged "better" by the interval alone instead of the multiple-comparison-corrected p-value. Each was fixed with a test that failed first (Step 27; `docs/decisions.md`, 2026-11-16). One-sentence version: "I had my own guardrails reviewed and closed every gap found, with a test proving each fix."

**"Isn't a version 2 just trying until something works?"** That is what it would be if I ran new models on the v0.1 test periods until one won. v0.2 does the opposite: exactly two variants, fixed by reasoning from the v0.1 diagnosis before any v0.2 result; a dataset no model has touched; the same headline and budget as before; a Holm correction across every comparison; the success rule written in advance; and a sealed, committed plan before scoring, with the test scored once. Every variant ever tried is counted, and a second negative gets reported the same way as a win (Steps 28 to 31, `docs/v2-plan.md`). One-sentence version: "Trying until something works means many tries on one test; I made two reasoned tries on a new test, sealed in advance, and promised to report a no."

**"Why should I trust v0.2 more than tweaking v0.1?"** Because tweaking v0.1 would mean choosing settings while looking at a test I had already scored, so any gain would be partly luck I picked. In v0.2 the settings are fixed before the new data is touched, development happens only on data that was already used, and git history shows the plan was committed before the first LI-Small score. It is the same protection that stopped my +5.7-point validation result from being reported as a win in v0.1. One-sentence version: "A tweak is graded on an exam I have already seen; v0.2 is graded on a new exam, with the answers I would accept written down first."
