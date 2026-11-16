# Good first issues

**Status: drafts, not posted.** Three proposed GitHub issues for outside contributors, taken from the next steps in `docs/findings.md` and the limitations register in `docs/methodology.md`. Each should be opened only after the owner reviews it.

Ground rules for every issue: read `CLAUDE.md` first. Splits go through `cleanroom.split`, numbers come from run files, any new choice gets a `docs/methodology.md` entry, and negative results stay on record. The AMLWorld test period has already been scored once under a sealed plan, so no issue below may score it again without a new lock, and any such run must be counted and reported as exploratory.

---

## 1. Compare the AMLWorld mirror with the official Kaggle file

**Motivation.** Tells reads AMLWorld HI-Small from a Hugging Face mirror because it needs no login. Its row and account counts match the published table (5,078,345 transfers, 515,088 accounts), but the file has not been compared byte for byte with the official Kaggle copy (methodology M8, limitations register L4). Until it is, every AMLWorld result rests on an unconfirmed source. This is a small, self-contained task that needs a free Kaggle account and no GPU.

**Acceptance criteria.**

- The official Kaggle `HI-Small_Trans.csv` is downloaded and its SHA-256 is computed.
- The result is recorded next to the mirror's checksum in `data/checksums.sha256`, with the source URL and download date.
- If the checksums differ, a short note reports whether row count, account count, laundering count, and date range still match, measured with `scripts/check_data_fit.py`, and whether the difference is only formatting (line endings, column order).
- `docs/methodology.md` M8 and L4 are updated to say what was found. No result is rerun as part of this issue.

**Pointers.** `data/README.md`, `data/checksums.sha256`, `src/tells/data/amlworld.py`, `scripts/check_data_fit.py`, `docs/methodology.md` (M8, L4), `docs/data-fit.md`.

---

## 2. Add an LLM review pass to the agent's Critic

**Motivation.** The Critic in `src/tells/agent/critic.py` checks the agent's claims deterministically: no typed numbers, every run exists and passed its canary and control, test runs were locked, and each claimed direction matches its bootstrap interval. It does not judge the reasoning itself, for example a narrative that overstates a descriptive hint as a finding. `CLAUDE.md` asks for deterministic checks first and an LLM review second; the second step is not built yet (methodology M40).

**Acceptance criteria.**

- A new review step runs only after all deterministic checks pass, and calls the model through the existing interface in `src/tells/agent/llm.py`.
- The reviewer may flag problems (overclaiming, a descriptive result presented as pre-registered, a missing caveat) but cannot clear a claim the deterministic checks rejected, and cannot add numbers to the report.
- Flags appear in the Reporter's draft next to the narrative they refer to.
- Tests replay a hand-written cassette covering at least one passing narrative and one overclaiming narrative, so they run free and without an API key.
- `docs/methodology.md` M40 is updated, including what the reviewer is and is not allowed to do.

**Pointers.** `src/tells/agent/critic.py`, `src/tells/agent/llm.py`, `src/tells/agent/reporter.py`, `src/tells/agent/runner.py`, `tests/test_agent_loop.py`, `tests/test_agent_tools.py`, `CLAUDE.md` ("Agent safety rules"), `docs/methodology.md` (M39, M40).

---

## 3. Rolling retrain on AMLWorld so training includes established accounts

**Motivation.** AMLWorld starts on 2022-09-01, so the fixed training window (Sept 1 to 4) holds almost only new accounts: 9.3% of training account-days have 16 or more earlier events, against 48.4% on validation and 62.7% on test (methodology M41, limitation L18). TSFM features are learned on short histories and graded mostly on long ones. A rolling retrain, where the model for each scored day is trained on all labeled days before it (minus the gap), would narrow this shift and is the third next step in `docs/findings.md`. M14 notes that the same code can support it.

**Acceptance criteria.**

- A config option for a rolling (expanding-window) retrain, with every window built through `cleanroom.split` and its one-day gap checked on the rows.
- Contestants B and D are retrained per scored day on identical rows, with B's fixed settings, and the history mix (`history_mix_share_16_plus`) is reported for each training window.
- The canary and label-permutation control run on every window, and the first-fraud-day headline is pooled across scored days with the existing account-level bootstrap.
- First run on the validation period only. Any test-period run needs a new lock, is counted in its uses file, and is reported as exploratory because that period has already been scored.
- `docs/methodology.md` gets a new entry (what, why, what else was considered, limitation) and L18 is updated with the measured effect, whether or not D improves.

**Pointers.** `configs/amlworld.yaml`, `packages/cleanroom/src/cleanroom/split.py`, `src/tells/harness.py`, `src/tells/config.py`, `src/tells/features/tsfm_features.py`, `docs/methodology.md` (M14, M15, M34, M41, L18), `docs/findings.md` (Section 8).
