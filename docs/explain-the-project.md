# Explaining Tells: the project step by step

This is the guide to read before talking about Tells with a recruiter, a hiring manager, or an interviewer. It tells the story in the order the work happened. Every step has the same parts:

- **What I did**
- **Why**
- **What else I considered**
- **Cause and effect**: what happened because of this step, including problems it exposed
- **How to say it**: one or two sentences you can say out loud

The technical detail behind each step is in `docs/methodology.md` (numbered M1, M2, ...), the dated diary is in `docs/decisions.md`, and the numbers come from run files under `artifacts/runs/`. This file links them into one story.

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
- **Cause and effect:** AMLWorld later turned out to have a training/evaluation history shift (Step 18), which only showed up because the two datasets differ.
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
- **How to say it:** "Any number in my README can be traced to the exact code, data, and model weights that produced it."

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
- **Cause and effect:** every control so far scored like random ranking, so no hidden leak. On full AMLWorld, TTM columns provide 16.5% of D's SHAP attribution.
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

### Step 23. Sealed test results

- **What I did:** sealed the AMLWorld plan (split, headline, contestants, TSFM settings) by hash, committed the lock file, then scored the test period (Sept 8 to 10) exactly once. The TabFormer plan was sealed at the same time, before any full-data TSFM run on it.
- **Result:** D (bank-style model plus TTM) caught 34.5% of newly bad accounts on their first bad day at a 0.5% budget, against 34.4% for B: +0.1 points, 95% interval -1.6 to +1.9, p 0.94, across 1,798 newly bad accounts. No difference.
- **Descriptive details (not pre-registered):** D was ahead at the tightest budget (13.6% against 11.0% at 0.1%) and on transaction-level ranking (PR-AUC 0.398 against 0.321), and behind at loose budgets (58.8% against 63.1% at 2%).
- **Cause and effect:** the validation gain (+5.7 points) came from a variant chosen after seeing the first validation run, on a single day. The sealed test, with 2.7 times as many new bad accounts and no choices made on it, did not confirm it. This is the protection the lock was built for, and it worked.
- **Integrity checks on this run:** canary passed; the label-permutation control scored like random ranking; test run number 1 under lock `f6ff26be`.
- **How to say it:** "My validation result looked like a win, but I had picked that version after seeing an earlier failure, so I sealed the plan before touching the test data. On the sealed test the foundation model added nothing at the budget I had committed to. That is my headline result, and I trust it more than the validation number because of how it was produced."

**Follow-up you will get: "So the project failed?"** No. The question was whether a TSFM adds value at a fixed review budget, under an evaluation that cannot be gamed. The answer, for TTM on AMLWorld at 0.5%, is "not detectably", with a tight interval (about plus or minus 2 points). A clean negative result from a rigorous setup is more useful to a fraud team than an unverified positive one, and the hints (better at the tightest budget, better transaction ranking) say where to look next.


---

## Part 8. The agent

### Step 24. An AI assistant that runs experiments but cannot cheat

- **What I did:** Claude Opus 5.5 in a loop I own, with three tools only (run an experiment by config name, read a run's summary, get chart data), a 10-call budget, and a rule that it may not write numbers. A Critic checks every claim against the run files without any model; a Reporter puts the model's words next to tables generated by code, marked as a draft for human review.
- **Why a loop I own:** every call must pass my allow-list and every exchange must be recordable for free, repeatable tests.
- **Why no numbers from the model:** language models can invent numbers; numbers come only from files.
- **How to say it:** "The agent can plan and run experiments, but it cannot read raw data, cannot open the sealed test without the lock, and cannot type a number. A checker rejects any claim the data does not support."

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
| pytest | 165 tests | Standard; supports skipping data-dependent tests in CI |
| ruff | Lint and formatting | One fast tool for both |
| GitHub Actions | CI on Python 3.11 and 3.12 | Free for the repo; every push is tested |
| Anthropic Python SDK | The agent's model calls | Official SDK for Claude; optional install so the rest needs no key |
| Hugging Face Hub | Model downloads | Where both TSFMs are published; pinned to exact commits |

---

## Part 10. Hard questions and honest answers

**"Isn't this all synthetic data?"** Yes, from two IBM generators. Real transaction data is private. I measured each dataset before using it, removed an artifact, and say everywhere that results do not show real-bank performance.

**"Your validation gain came after you changed something. Isn't that p-hacking?"** It was the second variant tried, chosen after the first; I recorded both, counted them, and sealed the plan before scoring the test period, which is the number that counts.

**"Why not a graph model? The best AMLWorld papers use the transaction network."** Out of scope: the question is whether account histories add to a bank-style model. My baseline has money-movement red flags so it is fair, but it is weaker than the best graph models, and I say so.

**"Why didn't you run the big model on everything?"** No GPU. I measured it at about 125 ms per account-day on CPU; full data would take days. I ran it on a sample and labeled that.

**"How do you know there is no leakage?"** Features are proven past-only by replay; every run plants two cheats that must be caught; a model trained on shuffled labels must score like random, and did; the test plan is sealed by hash before scoring.

**"What went wrong?"** My first TSFM result was negative because of a dataset shift and a scaling mistake; my first confidence intervals were biased; my first headline counted known-bad accounts; my first canary had a blind spot. Each was found by a check, fixed, tested, and documented.

**"What would you do next?"** Run Chronos-2 on full data with a GPU, test on real data under an agreement, and try a rolling retrain so the model sees established accounts during training.
