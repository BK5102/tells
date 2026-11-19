# Owner runbook: the three remaining manual steps

Tells v0.1 and v0.2 are finished. Three steps were deferred because they need something only the owner has: an Anthropic API key, a Kaggle account, or a person at the keyboard. Each one below is ready to run with one documented command. None of them scores a sealed test period or changes a result, a lock, or a sealed config.

Run everything from the repo root in Git Bash (Windows), or any bash shell. First install everything once:

```
uv sync --all-packages --extra tsfm --extra agent
```

## 1. A live agent run (needs ANTHROPIC_API_KEY)

What it shows: the agent (`docs/methodology.md` M40) plans and runs an experiment with real Claude calls, the Critic checks every claim against the run files, and the Reporter writes a draft. So far it has only been tested with a hand-written cassette (a saved conversation replayed in tests).

```
export ANTHROPIC_API_KEY=<your key>   # never commit it; .env is gitignored
uv run tells agent "On the TabFormer sample, does adding TTM help the bank-style model at the headline budget?" --cassette artifacts/agent_live_cassette.json --record
```

- It runs on the validation period only. The agent cannot touch a test period unless `--allow-test` is given; do not give it.
- It stops after 10 tool calls (`--max-tool-calls`). Each question costs money at Opus 5.5 prices (see the Claude API pricing page).
- Output: `artifacts/agent_report.md` (the draft report) and `artifacts/agent_live_cassette.json` (the recorded conversation, which replays for free with the same command minus `--record`). The command exits with 0 if the Critic passed and 1 if it rejected a claim; both are results worth keeping.
- Afterwards: add one dated note to `docs/decisions.md` and a sentence to `docs/explain-the-project.md` Step 26 saying a live run was made and what the Critic said. Copy no number by hand; point to the report file.

## 2. Compare the AMLWorld mirror with the official Kaggle file (needs a free Kaggle account)

What it shows: whether the Hugging Face mirror Tells used is byte-identical to the official file (`docs/methodology.md` M8, limitation L4, good first issue 1). The official dataset is "IBM Transactions for Anti Money Laundering (AML)", https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml.

Download `HI-Small_Trans.csv` (and `LI-Small_Trans.csv` for v0.2) from that page in a browser, or with the Kaggle command-line tool (`uv run --with kaggle kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f HI-Small_Trans.csv -p data/raw/amlworld_kaggle`, then unzip it if it arrives zipped). Put the files in `data/raw/amlworld_kaggle/` (gitignored). Then:

```
uv run python scripts/verify_checksums.py --entry raw/amlworld_hf_mirror/HI-Small_Trans.csv --file data/raw/amlworld_kaggle/HI-Small_Trans.csv
uv run python scripts/verify_checksums.py --entry raw/amlworld_li_hf_mirror/LI-Small_Trans.csv --file data/raw/amlworld_kaggle/LI-Small_Trans.csv
```

If you computed a checksum somewhere else (for example `sha256sum` on another machine), paste it instead of `--file`: `--sha256 <64 hex characters>`.

- `MATCH` (exit 0): the mirror is the official file. Record the Kaggle checksum, source URL, and download date in `data/checksums.sha256` and `data/README.md`, and update M8 and L4 to say the source is confirmed.
- `DIFFERENT` (exit 1): this alone does not mean the data differ, because line endings or column order change the bytes. Compare counts with `uv run --with polars python scripts/check_data_fit.py amlworld data/raw/amlworld_kaggle/HI-Small_Trans.csv` against `docs/data-fit.md`, and write down what you find in M8 and L4. Rerun no result.

To check that the local copies still match what was recorded: `uv run python scripts/verify_checksums.py` (about a minute; it hashes about 3.9 GB).

## 3. A terminal recording for the README

What it shows: in about two minutes (110 seconds measured on 2026-11-18), the Cleanroom tests, the leak and lock tests, a small experiment, its code-built report, and the sealed v0.2 headline.

```
bash scripts/demo.sh
```

It needs the prepared TabFormer sample (`uv run tells prepare tabformer`, see `data/README.md`). Record it with any screen recorder, or with asciinema on macOS, Linux, or WSL: `asciinema rec -c "bash scripts/demo.sh" docs/demo.cast`. It reuses the newest small run if one exists; `bash scripts/demo.sh --fresh` runs `configs/small.yaml` first, which took 7 to 10 minutes on a laptop CPU with the TTM cache filled. Run it once before recording so caches are warm. Then link the recording from the README "Quickstart" section.
