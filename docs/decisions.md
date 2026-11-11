# Decisions

One entry per decision or feature: what was decided, why, and an analogy. These are the owner's talking points. Newest at the bottom.

## Phase 0 (2026-11-10)

### D1. One Cleanroom package, shared by all three repos

The Cleanroom is written once, as its own installable Python package inside the Tells repo (`packages/cleanroom/`), and Reserve and Culprit install a tagged version of it from git instead of keeping their own copies. Copies drift: a leak bug fixed in one repo would stay alive in the other two, and the claim "all three projects sit the same honest exam" would stop being true. Analogy: three schools using the same sealed exam board, rather than each school photocopying the rulebook and editing its own version.

How siblings install it (after the `cleanroom-v0.1` tag exists):

```
uv add "cleanroom @ git+https://github.com/<owner>/tells@cleanroom-v0.1#subdirectory=packages/cleanroom"
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
