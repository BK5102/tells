# TSFM notes (Day 12 spike, 2026-11-14)

Every row is checked against the model's own code or documentation at build time. "Unknown" means not found, not guessed.

| Model | Checkpoint | License | Size | Embeddings available? (how) | Input type | Context | Pretraining data and cutoff | Source |
|---|---|---|---|---|---|---|---|---|
| Chronos-2 | `amazon/chronos-2` (chronos-forecasting 2.3.2) | Apache-2.0 | 119.5M parameters | **Yes, official API.** `Chronos2Pipeline.embed(inputs, batch_size, context_length)` returns, per series, a tensor of shape (n_variates, num_patches + 2, d_model): one embedding per patch plus a [REG] token and a masked output patch | Univariate or multivariate (variates share information through group attention); past-only, known-future, and categorical covariates | Trained up to 8,192 steps | Real univariate datasets (energy, traffic, weather, web, cloud traces, M4) plus synthetic univariate and multivariate data. **No financial or transaction data is listed.** Cutoff: **undisclosed** | `inspect.getdoc(Chronos2Pipeline.embed)`; Hugging Face model card; [Ansari et al. 2025](https://arxiv.org/abs/2510.15821) Table 6 |
| TTM r2 (Tiny Time Mixer) | `ibm-granite/granite-timeseries-ttm-r2` | Apache-2.0 | 0.8M parameters | To verify on Day 17: backbone hidden states through the `granite-tsfm` package | Multivariate with channel mixing | Fixed context per checkpoint (to verify) | To verify; cutoff undisclosed until checked | Hugging Face model card |
| Chronos-Bolt small | `amazon/chronos-bolt-small` | Apache-2.0 | 47.7M parameters | Not chosen: same family as Chronos-2, so it adds little variety | Univariate | | | Hugging Face API |

CPU timing for Chronos-2 embedding (laptop, 8 logical cores, measured 2026-11-14 while a LightGBM run shared the CPU, so treat as an upper bound), batches of 256 series:

| Input series per account | Events per account | Time per account | Output per account |
|---|---|---|---|
| 1 | 64 | 38 ms | 6 patch embeddings x 768 |
| 4 | 64 | 158 ms | 4 x 6 x 768 |
| 4 | 256 | 387 ms | 4 x 18 x 768 |
| 4 | 1,024 | 1,340 ms | 4 x 66 x 768 |

Consequence: the AMLWorld validation day alone has 224,616 account-days; at 4 input series and 256 events that is about a day of CPU, and training needs several times more. Full-data Chronos-2 embedding therefore runs once on a free GPU (Colab), with results cached on disk; everything else, including development on the samples and the second model (TTM, 0.8M parameters), runs on CPU. See methodology M32.

## Choice

**First TSFM: Chronos-2.** It is the model the AMLWorld paper used (so results can be set beside theirs), it has an official embedding API, it reads several parallel series of one account at once (amount, time gap, direction), and it is openly licensed. **Second TSFM: TTM r2**, chosen for contrast: about 150 times smaller, designed for CPU, from a different lab and architecture. If a 0.8M-parameter model matches a 119M one, size is not what helps; if neither helps, the result is stronger.

## Contamination note (methodology M23, PLAN FR15)

TabFormer was released in 2020 and AMLWorld in 2023, both before Chronos-2. Neither appears in Chronos-2's documented training data, which lists no financial data at all, but the cutoff is undisclosed and documentation may be incomplete, so contamination cannot be fully ruled out. Mitigations: the embeddings are frozen and only feed contestant D, the canary and time-shuffled control run on every experiment, and the test period is scored once under the lock.

## How account histories become model input (Day 13 design)

One sequence per account, one step per event, oldest first, as in [Savvidi et al. 2026](https://arxiv.org/html/2610.29667v1): parallel channels for amount (log of absolute USD), time since the previous event (log minutes), direction (+1 out, -1 in, 0 self), and, for TabFormer, an online flag. One sequence per account-day case: only events strictly before midnight of that day enter it (methodology M32), which is stricter than the checklist's per-transaction rule and is tested with `cleanroom.pastonly`. Context is capped (default 256 events) to keep CPU cost bounded; the cap is an ablation on Day 26.
