# TSFM notes (Day 12 spike, 2026-11-14)

Every row is checked against the model's own code or documentation at build time. "Unknown" means not found, not guessed.

| Model | Checkpoint | License | Size | Embeddings available? (how) | Input type | Context | Pretraining data and cutoff | Source |
|---|---|---|---|---|---|---|---|---|
| Chronos-2 | `amazon/chronos-2` (chronos-forecasting 2.3.2) | Apache-2.0 | 119.5M parameters | **Yes, official API.** `Chronos2Pipeline.embed(inputs, batch_size, context_length)` returns, per series, a tensor of shape (n_variates, num_patches + 2, d_model): one embedding per patch plus a [REG] token and a masked output patch | Univariate or multivariate (variates share information through group attention); past-only, known-future, and categorical covariates | Trained up to 8,192 steps | Real univariate datasets (energy, traffic, weather, web, cloud traces, M4) plus synthetic univariate and multivariate data. **No financial or transaction data is listed.** Cutoff: **undisclosed** | `inspect.getdoc(Chronos2Pipeline.embed)`; Hugging Face model card; [Ansari et al. 2025](https://arxiv.org/abs/2510.15821) Table 6 |
| TTM r2 (Tiny Time Mixer) | `ibm-granite/granite-timeseries-ttm-r2`, context 512, prediction 96 (granite-tsfm 0.3.12) | Apache-2.0 | 0.8M parameters | **Yes.** The forward pass returns `backbone_hidden_state` of shape (batch, channels, patches, 192) and `prediction_outputs` of shape (batch, 96, channels); `past_observed_mask` marks padding | Multivariate | Fixed 512-step window | Cutoff undisclosed (not checked in a paper yet) | `get_model` output keys inspected 2026-11-14; Hugging Face model card |
| MOMENT-1-small (v0.2) | `AutonLab/MOMENT-1-small`, revision `411e288267f82cce86296dbe4d6c8bc533cc162f` (momentfm 0.1.4) | MIT | 35.3M parameters (counted after loading) | **Yes.** `MOMENTPipeline.from_pretrained(..., model_kwargs={"task_name": "embedding"})`, then `init()`; the forward pass with `reduction="none"` returns (batch, channels, 64 patches, 512); the default `reduction="mean"` averages channels and real patches into one 512 vector. A patch is real only if all 8 steps are real. No pretrained forecast head | Multichannel, each channel encoded separately | Fixed 512-step window, 8-step patches, `input_mask` marks real steps | The Time Series Pile (public collection); financial content not checked | Package source `momentfm/models/moment.py`, `utils/masking.py` read 2026-11-17; Hugging Face API; [Goswami et al. 2024](https://arxiv.org/abs/2402.03885) |
| Chronos-Bolt small | `amazon/chronos-bolt-small` | Apache-2.0 | 47.7M parameters | Not chosen: same family as Chronos-2, so it adds little variety | Univariate | | | Hugging Face API |

CPU timing for Chronos-2 embedding (laptop, 8 logical cores, measured 2026-11-14 while a LightGBM run shared the CPU, so treat as an upper bound), batches of 256 series:

| Input series per account | Events per account | Time per account | Output per account |
|---|---|---|---|
| 1 | 64 | 38 ms | 6 patch embeddings x 768 |
| 4 | 64 | 158 ms | 4 x 6 x 768 |
| 4 | 256 | 387 ms | 4 x 18 x 768 |
| 4 | 1,024 | 1,340 ms | 4 x 66 x 768 |

Consequence: the AMLWorld validation day alone has 224,616 account-days; at 4 input series and 256 events that is about a day of CPU, and training needs several times more. The plan was to run full-data Chronos-2 once on a free GPU; no GPU was available, so it ran only on the 20,000-account AMLWorld sample, with results cached on disk; everything else, including development on the samples and the second model (TTM, 0.8M parameters), runs on CPU. See methodology M32.

## Choice

**First TSFM: Chronos-2.** It is the model the AMLWorld paper used (so results can be set beside theirs), it has an official embedding API, it reads several parallel series of one account at once (amount, time gap, direction), and it is openly licensed. **Second TSFM: TTM r2**, chosen for contrast: about 150 times smaller, designed for CPU, from a different lab and architecture. If a 0.8M-parameter model matches a 119M one, size is not what helps; if neither helps, the result is stronger.

## Contamination note (PLAN FR15; see audit #2, check 9)

TabFormer was released in 2020 and AMLWorld in 2023, both before Chronos-2. Neither appears in Chronos-2's documented training data, which lists no financial data at all, but the cutoff is undisclosed and documentation may be incomplete, so contamination cannot be fully ruled out. Mitigations: the embeddings are frozen and only feed contestants C and D, the canary and the label-permutation control run on every experiment, and the test period is scored once under the lock.

## How account histories become model input (Day 13 design)

One sequence per account, one step per event, oldest first, as in [Savvidi et al. 2026](https://arxiv.org/html/2610.29667v1): parallel channels for amount (log of absolute USD), time since the previous event (log minutes), direction (+1 out, -1 in, 0 self), and, for TabFormer, an online flag. One sequence per account-day case: only events strictly before midnight of that day enter it (methodology M32), which is stricter than the checklist's per-transaction rule and is tested with `cleanroom.pastonly`. Context is capped (default 256 events) to keep CPU cost bounded; varying the cap was planned as an ablation but not run (each needs a new embedding run on CPU).

## CPU cost on real account histories (2026-11-14, no other load)

Measured on 512 real account-day histories from each development sample:

| Model | AMLWorld (median 10 events) | TabFormer (256 events) |
|---|---|---|
| Chronos-2 embedding | 66 ms per account-day | 283 ms |
| TTM embedding and forecast | 2.3 ms | 2.9 ms |

Account-days per run: AMLWorld sample 96,177 (21% with no history), TabFormer sample 558,323. What this allows on CPU is in methodology M36.

## v0.2 CPU cost (2026-11-17, shared machine)

Measured on 512 synthetic histories with realistic lengths (median 10 events, 90th percentile 37, maximum 256), 2 CPU threads while other jobs were running, best of 3 runs. Treat as upper bounds.

| Model | Time per account-day | Embedding size before PCA |
|---|---|---|
| TTM (v0.1, mean pooling, reference) | 2.35 ms | 576 |
| `ttm_recent` (TTM, last patch only) | 2.50 ms | 576 |
| `moment` (MOMENT-1-small) | 143 ms | 1,536 |

MOMENT's cost does not depend on history length (every window is 512 steps), so at this rate the 20,000-account AMLWorld sample (96,177 account-days) needs about 4 CPU hours and one full AMLWorld day about 9. Details and choices: methodology M45 and M46.
