# CLAUDE.md: Tells

Read this first in every session. It tells you what this project is, how it is laid out, and the rules you must not break.

## What this project is

Tells tests whether a time series foundation model (TSFM) can improve fraud and money-laundering detection compared with the checklist-plus-tree-model approach banks use today, under an evaluation built so the model cannot cheat. An AI agent runs the experiments and drafts the audit report. See `PLAN.md` for diagrams, requirements, and the 30-day plan.

Plain-English summary: banks can only review a small share of suspicious transactions, so the question is who lands at the top of the list. Tells measures whether adding a pretrained "behavior fingerprint" of each account helps, and proves the test was fair.

## Who you are working with

The owner is a data science and economics student who is new to finance and to TSFMs. Explain every new term in one plain sentence with a real-world analogy the first time it appears. Keep messages short and direct.

Writing style for all docs, comments, commit messages, and chat:

- Concise, specific, plain language.
- No em-dashes. No emojis.
- Prose over bullet points when prose reads fine.
- Make edits precisely. Do not rewrite surrounding content that was not asked about.
- Cite sources for factual claims about the field. If you cannot source it, say so.

## Repo layout

```
tells/
  CLAUDE.md  PROMPT.md  PLAN.md  README.md
  pyproject.toml  uv.lock          # uv workspace root; members: packages/cleanroom
  packages/cleanroom/              # the shared Cleanroom, its own installable package
    pyproject.toml
    src/cleanroom/  split.py  lock.py  metrics.py  calibration.py  bootstrap.py
                    stats.py (multiple-comparison correction)  canary.py  provenance.py
    tests/
  src/tells/
    data/        loaders for each dataset, one file per dataset
    cases.py     builds account-day cases from transactions
    features/    checklist.py  embeddings.py  surprise.py
    models/      contestants.py  tsfm/ (one adapter per TSFM)
    agent/       planner.py  runner.py  critic.py  reporter.py  tools.py
    cli.py
  tests/
  scripts/       check_data_fit.py and download scripts
  configs/       experiment YAML files
  docs/          tsfm-notes.md  decisions.md  data-fit.md
  data/          gitignored except README.md and checksums.sha256
  artifacts/     gitignored except small sample outputs
```

The Cleanroom (`packages/cleanroom`, imported as `cleanroom`) is shared with the sibling projects Reserve and Culprit, which install a tagged version from git (see `docs/decisions.md` D1). Rules for it: no fraud, credit, or payments code or vocabulary; it never imports from `tells`; every change that siblings need is released as a new `cleanroom-vX.Y` git tag; breaking changes bump the minor version and are noted in `packages/cleanroom/CHANGELOG.md`.

## Data and scoring unit

Measured fit for every candidate dataset is in `docs/data-fit.md`. Primary dataset: IBM TabFormer credit card transactions. Second: AMLWorld HI-Small, truncated to 2022-09-01 through 2022-09-10, because the data after that is about 60% laundering, a generator artifact. Optional: Sparkov. IEEE-CIS and PaySim are not used (no usable account histories; see the ranking in `docs/data-fit.md`).

The review budget applies to account-day cases (one case per account per active day; score is the max transaction score that day; label is 1 if any transaction that day is positive), as a share of each day's cases. Transaction-level PR-AUC is reported as a secondary number. See `docs/decisions.md` D3.

## Commands

```
uv sync --all-packages                    # install tells and cleanroom
uv run pytest -q                          # tests for tells and cleanroom
uv run --with polars python scripts/check_data_fit.py tabformer data/raw/tabformer/card_transaction.v1.csv
uv run ruff check . && uv run ruff format --check .
uv run tells run configs/small.yaml       # small-subset experiment, CPU, under 15 minutes
uv run tells agent "does a longer history window help?"
```

If a command above does not exist yet, create it as part of the day's task and update this file.

## Rules you must not break

1. **No future information.** Every feature at time t uses only rows with time less than t. Every feature function has a unit test that proves it.
2. **All splits go through `cleanroom.split`.** Never write an ad hoc train/test split anywhere else.
3. **No test scoring without a lock.** The sealed-envelope lock (config hash recorded before test scoring) must exist or scoring raises an error. Never bypass it, even for debugging.
4. **Report numbers come from result files.** The reporter inserts numbers programmatically. An LLM may write narrative only, and must reference numbers, not type them.
5. **No real customer data or personal information.** Public or synthetic datasets only.
6. **No secrets in git.** API keys go in environment variables. Add `.env` to `.gitignore`.
7. **Record provenance.** Each result stores config hash, git commit, model name and version, and data checksum.
8. **Leak tests first.** When adding a feature, write the leak test before the feature.
9. **Never invent results, datasets, or citations.** If a number is unknown, leave a clearly marked placeholder.
10. **Negative results stay in.** Do not hide runs where the TSFM did not help.
11. **Correct for many comparisons.** When more than one contestant or variant is compared, apply the Cleanroom's multiple-comparison correction and report how many variants were tried.
12. **Past labels are not free.** A fraud label is only known after review. Never use an account's earlier labels as a feature unless a realistic label delay is applied and tested.

## Agent safety rules (for the agent code in `src/tells/agent/`)

- Tool allow-list only: `run_experiment`, `load_results`, `plot`. No shell, no network, no arbitrary file access.
- The Runner cannot read test labels before the lock exists.
- The Critic runs deterministic checks first (split order, lock present, canary caught, interval width) and an LLM review second.
- LLM calls sit behind an interface. Tests use recorded cassettes so they are deterministic and free.

## Workflow

- One shippable feature per day, as listed in `PLAN.md` Part 6. Small commits, each with a test.
- Commit style: `type: short summary` (for example `feat: add past-only rolling count features`). Use `feat`, `fix`, `test`, `docs`, `chore`.
- Before finishing a task: run tests and lint, then state what changed and what you did not verify.
- When a task is ambiguous, ask one specific question instead of guessing.
- After each feature, add a 3-sentence plain-English note to `docs/decisions.md` explaining what you built and why, with an analogy. These notes become the owner's talking points.

## Definition of done for any feature

Tests pass, lint passes, leak test exists if it touches features or splits, docs updated if behavior changed, and the change is explained in plain English.
