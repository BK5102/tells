# Leak audit #2: the TSFM pipeline (2026-11-16)

Run with the leak audit prompt in `PROMPT.md` section 3, after contestants C and D, the TSFM cache, the label-permutation control, and the agent tools were added. Each check is PASS, FAIL, or WARN (works, with a caveat to fix or watch). Evidence points to code, tests, and run files.

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Do TSFM inputs use anything at or after the scored day? | PASS | Each account-day's sequence holds only events strictly before midnight that day (`src/tells/features/sequences.py`, `SequenceIndex.build`). `tests/test_sequences.py` proves it with `cleanroom.pastonly`, and a planted bug (cut at the end of the day) failed two tests. |
| 2 | Do TSFM features, including the surprise score, use the future or labels? | PASS | `tests/test_tsfm_pipeline.py::test_tsfm_features_are_past_only` runs the whole path (checklist, sequences, model, surprise) through `assert_past_only` and `assert_ignores(label)`. Model inputs are only three channels: log amount, log gap, direction (`CHANNELS`); no label channel exists. |
| 3 | Is PCA fitted on anything but training days? | PASS | `fit_pca` uses only account-days inside the training window, for the mean, the scale, and the components (`src/tells/features/tsfm_features.py`). `test_pca_uses_training_days_only` multiplies every later amount by 1,000 and confirms the PCA does not move. |
| 4 | Could a cached embedding come from different data, code, or model weights? | PASS | The cache key covers model version, data checksum, sequence-code checksum, and context; the cache also records the model revision and refuses a different one (`test_cache_resumes_and_is_keyed_by_data`, `test_cache_refuses_a_different_model_revision`). Both models are pinned to exact Hugging Face commits. |
| 5 | Does the whole pipeline leak labels somewhere we did not think of? | PASS | The label-permutation control (B retrained on labels shuffled within each day) scored like random ranking in every run: for example -0.2 points [-0.5, +0.2] against random on full AMLWorld (`controls` in each `results.json`). |
| 6 | Does the "first fraud day" answer key read labels it should not? | PASS | It uses labels strictly before the evaluated window's start only to exclude accounts already known bad (`src/tells/harness.py`), never as a model input. |
| 7 | Has the test period been scored, or a config sealed? | PASS | No `locks/` folder exists, and none of the 14 logged runs is a test-period run (`artifacts/runs/index.jsonl`). |
| 8 | Can the agent read labels or raw data? | PASS | `tests/test_agent_tools.py`: only three tools, results never contain rows, path tricks and unlocked test scoring are denied. |
| 9 | Could the TSFMs have seen these datasets in pretraining? | WARN | Chronos-2's documented training data lists no financial data; TTM's corpus was not checked in a paper; both cutoffs are undisclosed (`docs/tsfm-notes.md`). The embeddings are frozen and only feed D, and the test period is scored once, which limits but does not remove this. |
| 10 | Were choices made after looking at validation results? | WARN | Yes, one: standardizing embeddings before PCA, after the first TTM run made D worse. It is a standard step and the first run is kept, but the validation gain (+5.7 points) was produced by the second variant tried. This is why the test period must be sealed before any further change, and why the number of variants (2 on full AMLWorld) is reported. |
| 11 | Does training look like evaluation? | WARN | On AMLWorld, no: 9.3% of training account-days have 16 or more earlier events, against 48.4% on the validation day and 62.7% in the test period (methodology M41). Reported in every run as `history_mix_share_16_plus`. TabFormer has no such shift (99% in both). |

## Before sealing

1. Freeze the TSFM settings as they stand (standardized PCA, 16 components, context 256) and the contestant list, and seal the AMLWorld config with `tells lock`. No further validation-driven changes after that.
2. State in the sealed note how many variants were tried (2 on full AMLWorld), so the test result is read with that in mind.
3. Decide whether the TabFormer test runs on the 300-card sample (CPU-feasible, small) or on full TabFormer with TTM (about 9 CPU hours).
