# Tells

**Do pretrained time series models help catch fraud when analysts can only review a fixed number of cases?** Tells answers that with an evaluation built so the model cannot cheat, and an AI agent that runs the experiments and drafts an audit report.

In poker, a "tell" is a small behavior that gives away a hidden hand. A stolen card gives off tells too: odd timing, odd amounts, a sudden change in rhythm. Tells tests whether a time series foundation model (TSFM) can read those tells from an account's history, on top of the checklist-plus-tree-model approach banks use today.

> Status: early build. No results exist yet. Every number in this README that describes data was measured in `docs/data-fit.md`; there are no model results to report.

## The idea in one diagram

```mermaid
flowchart LR
    subgraph C1["PART A: WHAT IS IT?"]
        direction TB
        A["<b>1. TIME SERIES</b><br/>Numbers recorded in order over time<br/><i>Like: your phone's daily step count,<br/>or the line on a heart monitor</i>"]
        B["<b>2. FORECASTING</b><br/>Use the past to guess the next values<br/><i>Like: a weather forecast.<br/>Past temperatures in, tomorrow's temperature out</i>"]
        C["<b>3. BASELINE</b><br/>The laziest sensible guess, e.g. tomorrow = today<br/><i>Like: guessing 'it will be like yesterday'.<br/>A fancy model only counts if it beats this</i>"]
        D["<b>4. FOUNDATION MODEL (TSFM)</b><br/>One big model pre-trained on millions of different<br/>time series, then pointed at a brand new one<br/><i>Like: a doctor who has read a million patient charts,<br/>versus one who only ever studied a single patient</i>"]
        A --> B
        B --> C
        C --> D
    end
    subgraph C2["PART B: HOW DO YOU USE IT, AND HOW DO YOU GRADE IT?"]
        direction TB
        E{"<b>5. THREE WAYS TO USE IT</b>"}
        E1["<b>Zero-shot</b><br/>Ask it cold, no extra training<br/><i>Like: asking the doctor about a new patient<br/>with no extra briefing</i>"]
        E2["<b>Fine-tune</b><br/>Give it a short extra course on your data<br/><i>Like: a week of specialty training</i>"]
        E3["<b>Frozen embeddings</b><br/>Keep only its summary of the data and feed<br/>that summary to a simpler model<br/><i>Like: using the doctor's written notes as<br/>inputs to your own checklist</i>"]
        F["<b>6. EVALUATION</b><br/>Grading the model: how good is it, really?<br/><i>Like: an exam. The exam is only fair if<br/>the student has not seen the answers</i>"]
        G["<b>CHEAT 1: LEAKAGE</b><br/>The model is shown information from the future<br/><i>Like: a student who saw the answer key</i>"]
        H["<b>CHEAT 2: CONTAMINATION</b><br/>The test data was already inside the model's<br/>giant pre-training dataset<br/><i>Like: the exam questions were printed in the textbook</i>"]
        I["<b>HONEST EXAM</b><br/>Test only on dates AFTER every model's training cutoff.<br/>Beat the baselines. Check the win is not luck.<br/>Measure what it is worth in real decisions.<br/><i>Like: a sealed exam written after class ended</i>"]
        E --> E1
        E --> E2
        E --> E3
        E1 --> F
        E2 --> F
        E3 --> F
        F --> G
        F --> H
        G --> I
        H --> I
    end
    subgraph C3["PART C: WHO RUNS THE TEST?"]
        direction TB
        J["<b>7. AI AGENTS</b><br/>An LLM that uses tools in a loop to finish multi-step work<br/><b>Harness</b> = the lab bench and rulebook it works inside<br/><b>Orchestration</b> = a manager handing steps to specialist workers<br/><i>Like: a careful intern with a checklist who runs the<br/>experiments overnight. You review the write-up</i>"]
        K["<b>THIS REPO'S JOB</b><br/>Use an agent to run the honest exam on a<br/>real-money problem and produce a report<br/>a skeptical person could audit"]
        J --> K
    end
    C1 --> C2
    C2 --> C3
    classDef concept fill:#e8f1ff,stroke:#2b6cb0,color:#102a43;
    classDef warn fill:#ffe8e8,stroke:#c53030,color:#4a1010;
    classDef good fill:#e6f7ec,stroke:#2f855a,color:#0f3d24;
    classDef goal fill:#fff4d6,stroke:#b7791f,color:#4a3200;
    class A,B,C,D,E,E1,E2,E3,F,J concept;
    class G,H warn;
    class I good;
    class K goal;
```

More diagrams (the Cleanroom, fraud from zero, the full system) are in [`PLAN.md`](PLAN.md) and [`diagrams/`](diagrams/).

## What is being compared

Four contestants, ranked on the same time-ordered test period and the same daily review budget (for example the top 1% of each day's account-day cases):

| | Contestant | What it is |
|---|---|---|
| A | Lazy baseline | Rank by one simple rule |
| B | Checklist + tree model | Hand-built red-flag features into LightGBM, roughly what banks do today |
| C | TSFM only | Ranking from the pretrained model's view of each account |
| D | Checklist + TSFM embedding + surprise score + tree model | Contestant B with the TSFM's summary of each account added as extra columns |

## Data

Public synthetic data only. No real customer data.

| Dataset | Why it is here |
|---|---|
| [IBM TabFormer](https://github.com/IBM/TabFormer) credit card transactions | 6,139 cards from 1991 to 2020, with a median of 1,493 transactions before a card's first fraud. Long histories to read. |
| [AMLWorld](https://arxiv.org/abs/2306.16424) HI-Small | Money laundering with short histories (median 9 prior events), and the benchmark a published TSFM paper used. |

Why these two, and why IEEE-CIS and PaySim were set aside: [`docs/data-fit.md`](docs/data-fit.md).

## The Cleanroom

The evaluation rules live in `packages/cleanroom`, a separate, domain-neutral package that the sibling projects install from a git tag:

- time-ordered splits with a gap between train and test
- a sealed-envelope lock: the test plan is hashed before any test score exists, and scoring refuses to run without it
- a planted-leak canary that the harness must catch
- bootstrap intervals and multiple-comparison correction, so a lucky win does not look like a real one

## Quickstart

```
uv sync --all-packages
uv run pytest -q
uv run tells --version
```

The data and experiment commands are added as they are built; see `CLAUDE.md` for the current list.

## Repo layout

```
packages/cleanroom/   shared evaluation toolkit (its own package)
src/tells/            data loaders, features, contestants, agent
tests/                tests for tells (cleanroom tests live with the package)
scripts/              data-fit check and download helpers
docs/                 decisions.md, data-fit.md, tsfm-notes.md
data/                 README and checksums only; raw data is gitignored
```

## Limits

- Synthetic data is cleaner than reality. Results here will not show how a model performs on a real bank's data.
- This is not a production fraud system, and the agent never decides who is fraudulent. It runs experiments and drafts reports that a person reviews.
- Results will include the runs where the TSFM did not help.

## More

- [`PLAN.md`](PLAN.md): background from zero, requirements, and the 30-day plan
- [`docs/decisions.md`](docs/decisions.md): every design decision in plain English
- [`CLAUDE.md`](CLAUDE.md): rules for working in this repo
