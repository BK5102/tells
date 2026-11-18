# How an expert would review Tells

Written 2026-11-17, after all three sealed tests. It asks one question: if a fraud-detection or forecasting researcher read this project, what would they point out? It changes no result and runs no new experiment. Every number here is already in `docs/findings.md`.

## Short answer

The caveats listed with the v0.2 result would not count against the project. They are stated next to the result in every document, and in research, stating the limits of your own result is the expected standard, not a weakness. Those caveats are:

- one synthetic dataset;
- a modest effect;
- validation pointing the other way;
- the gain is specific to one model;
- the combined model was behind at loose budgets.

An expert would raise a few further points that the project does not answer yet. They are listed below, most important first, with an answer you can give. None of them means a rule was broken or a number is wrong. They are the questions that decide how far the result reaches.

## Already disclosed (an expert would see these as handled)

| Point | Where it is answered |
|---|---|
| Synthetic data only | README Limitations, methodology L1 and L25, guide Part 10 |
| No graph features, weaker than the best AMLWorld models | Methodology M18 and L7, guide Part 10 |
| Validation pointed the other way in v0.2 | Findings Section 10, L26, guide Step 33 |
| Gain is specific to Chronos-Bolt, not "TSFMs" | L27, findings Section 10 |
| Behind at loose budgets | L28, findings Section 10 |
| Many comparisons | Holm within each test, plus a Bonferroni bound across all four sealed "B + TSFM" tests (0.032) |
| Choices made after looking | Every variant counted; plans sealed before scoring; v0.1 validation variants listed |
| CPU-only limits (Chronos-2 on a sample, MOMENT dropped) | M36, M47, L22, L23 |
| Control failure on TabFormer validation | M42, L24, guide Part 10 |

## Not yet answered (what an expert would likely ask)

### 1. Is the gain from pretraining, or just from showing the model the raw history?

**The question.** Contestant D gets about 64 more columns describing each account's recent amounts, gaps and directions. A skeptic would ask whether the pretrained model matters at all. Simple summaries of the same history, such as recent quantiles, counts and trends, might give the same gain. So might the same Chronos-Bolt network with random, untrained weights.

**Why it matters.** This is the difference between "a foundation model helps" and "more history features help".

**Status.** Not tested.

**How it would be tested.** A pre-registered ablation on a new sealed test, with three versions:
- D with hand-made history summaries of the same three channels;
- D with an untrained network of the same shape;
- D with real Chronos-Bolt.

**How to say it.** "The result shows that adding Chronos-Bolt's summary helped. It does not yet show that the pretraining is what helped. The next test compares it with simple history statistics and with an untrained copy of the same network."

### 2. Is LI-Small an independent replication?

**The question.** LI-Small is fresh in the sense that no model or choice touched it. But it comes from the same IBM simulator as HI-Small, with a lower laundering rate. So a Chronos-Bolt gain could reflect how that simulator builds laundering patterns, not how real laundering behaves.

**Status.** Partly disclosed. L25 says "one synthetic dataset", and guide Step 30 says "same generator". This page states it plainly: v0.2 is a new draw from the same world, not a second world.

**How to say it.** "The fresh test rules out tuning to the test. It does not rule out quirks of the simulator, because both datasets come from the same generator."

### 3. Is the uncertainty interval too narrow?

**The question.** The 95% interval comes from resampling accounts, so it reflects which accounts landed in the test. It does not include two other sources of variation:
- **Training randomness.** LightGBM uses random row and column subsampling (80% each) with one fixed seed, so B and D were each trained once. A different seed would move both scores a little.
- **Day-to-day variation.** The test has three days. Resampling accounts does not capture a day that happens to be unusual.

**Why it matters.** With these sources included, the interval would be wider. Whether +1.1 points, the low end, would stay above zero is unknown.

**Status.** Not measured.

**How it would be tested.** Retrain B and D over several seeds, report the spread, and add a resampling step that resamples days as well as accounts.

**How to say it.** "My interval covers which accounts were in the test, not model-training randomness or day-to-day variation. With one training seed and three test days, the real uncertainty is somewhat larger than the interval shows."

### 4. Why does Chronos-Bolt help when TTM does not?

**The question.** An expert would want a mechanism, not just a difference. There are three candidates:
- Chronos-Bolt is trained to forecast with quantiles;
- it summarizes each channel separately;
- its patches are shorter.

**Status.** Not tested (L25 and the M49 caveats). The v0.1 diagnostics explain why TTM did not help, mainly that it is blind to the same day's transactions. They do not explain why Chronos-Bolt did.

**How to say it.** "I know which model helped, not why. That would need an ablation that changes one property at a time."

### 5. Would the gain matter to a real team?

**The question.** In operations, the question is cost and benefit, not percentage points. At a 0.5% budget, D catches about 50 more of 1,549 newly bad accounts. Against that are about 4 ms of CPU per account-day and a model that needs a separate embedding pipeline.

**Status.** The CPU cost is measured (`docs/tsfm-notes.md`). The money value is not estimated, because the data is synthetic and has no loss amounts tied to real outcomes.

**How to say it.** "Operationally it is about 50 more catches per 1,549 at the same analyst workload, for a few milliseconds of CPU per account per day. Whether that pays depends on loss sizes I can't measure on synthetic data."

## What this means for the project

By the standard of a careful first study, the project holds up:
- the question was fixed before testing;
- negative results were reported;
- tests were sealed before scoring and scored once;
- leak checks and controls ran on every run;
- limits are written next to the results.

The open points above are the normal next steps for a positive result, and an expert would expect them in a follow-up, not in the first study. In order, the follow-up would be:
1. Test whether pretraining matters (point 1).
2. Measure training-seed and day-level variation (point 3).
3. Replicate on data from a different generator or on real data under an agreement (point 2).

**How to say it:** "I pre-registered the test, scored it once, and wrote the caveats next to the result. The main things an expert would still ask are whether the gain comes from pretraining or just from more history features, and how much training randomness adds to the uncertainty. Those are my next experiments."
