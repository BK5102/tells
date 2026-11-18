# cleanroom

A small, domain-neutral toolkit for honest evaluation of models on time-ordered data. It is built inside the [Tells](https://github.com/BK5102/tells) repo and shared with the sibling projects Reserve and Culprit.

Install a tagged release from another project (0.2.0 is tagged `cleanroom-v0.2`; until that tag exists, use `cleanroom-v0.1`, which lacks the 0.2 modules):

```
uv add "cleanroom @ git+https://github.com/BK5102/tells@cleanroom-v0.2#subdirectory=packages/cleanroom"
```

Modules:

| Module | What it does | Why it exists |
|---|---|---|
| `split` | Time-ordered train / validation / test windows with a minimum gap, re-checked on the actual rows | A model is used on the future, so it must be graded on the future |
| `pastonly` | Rebuilds features from strictly earlier rows and requires an exact match; shuffles outcomes and requires no change | Proves features cannot see the future or the answer |
| `canary` | Plants a future-value and a future-label feature and requires both to be caught | Proves the leak checks themselves work on this data |
| `lock` | Seals a config by hash before any test score exists; counts every test run | Pre-registration: the plan cannot be changed after seeing results |
| `metrics` | Recall and precision at a per-group review budget (ties share slots), average precision | Rare positives and limited review capacity make accuracy meaningless |
| `calibration` | Brier score, reliability table, calibration error | A risk score that says 10% should be right about 10% of the time |
| `bootstrap` | Paired cluster bootstrap for the difference between two contestants | A win must be bigger than luck, and related rows must be resampled together |
| `stats` | Holm and Benjamini-Hochberg corrections | Testing many variants makes some look good by chance |
| `provenance` | Config hash, git state, data checksum, package versions | Every number must trace back to exact inputs |
| `guard` | A sealed run needs a matching lock, a clean git tree, and a committed lock; it logs "started" and "finished" | The run's commit must name the code that ran, and every attempt must be counted |
| `vault` | Seals an answer key by checksum and releases it only after a sealed run has finished | Predictions are final before anyone sees the answers |
| `forecast` | Pinball loss, weighted quantile loss, interval coverage with a Wilson interval, interval score, relative error | A forecast that gives a range should be scored on the range, not just its middle |
| `stats.diebold_mariano` | Diebold-Mariano test with the Harvey, Leybourne and Newbold correction | Two forecasters' errors over time are correlated, so an ordinary test overstates a win |
| `split.rolling_origins` | A series of expanding-window splits stepping forward in time | One backtest date can be lucky; many origins show whether a model is steadily better |
| `asof` | Joins each query time to the latest value released before it | A backtest must use figures as they were first published, not as later revised |
| `events` | Matches alarms to event intervals (detected, delay, false alarms) and sets a threshold for a target false alarm rate, with persistence | A detector is judged by how fast it catches events at an alarm rate people will tolerate |

Rules: no domain-specific code or vocabulary, no imports from any project package, and every change other projects need ships as a new `cleanroom-vX.Y` tag. See `CHANGELOG.md`.
