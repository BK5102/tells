# START HERE

Three projects, one shared niche: **time series foundation models (TSFMs) for fintech risk, tested honestly, with AI agents doing the repetitive work.**

| Project | Fintech area | One-line pitch | Name meaning |
|---|---|---|---|
| **Tells** | Fraud and risk | Do pretrained time series models help catch fraud at a fixed review budget? | A poker "tell" is a small behavior that gives away a hidden hand |
| **Reserve** | Credit risk and underwriting | Can they forecast when groups of loans go bad, even in a recession? | A loss reserve, and a wine that ages well (loans age in vintages) |
| **Culprit** | Payments intelligence | When approval rates drop, can an agent find which bank, card range, or gateway is to blame? | Someone has to find the culprit |

## Order to build

1. **Tells first.** Days 1 to 11 build the **Cleanroom**, the shared testing toolkit (time-ordered splits, sealed test plan, leak canary, luck checks, multiple-comparison correction). It lives in the Tells repo as its own package, `packages/cleanroom`, and is released by git tag (`cleanroom-v0.1`). Reserve and Culprit install that tagged version from git; they never keep their own copy. See `tells/docs/decisions.md` D1.
2. **Then Reserve or Culprit.** Reserve uses your Economics minor most. Culprit is the most agent-heavy.
3. You do not need to finish all three. One finished project beats three half-finished ones.

## Each folder contains

- `CLAUDE.md`: the rules and context Claude Code reads every session.
- `PROMPT.md`: copy-paste prompts (kickoff, daily, audit, quiz you, recruiter rehearsal).
- `PLAN.md`: diagrams from zero with real-world analogies, glossary, how to explain it out loud, requirements, 30-day one-feature-a-day plan, risks, sources.

The `diagrams/` folder has the same diagrams as images. The diagrams inside each `PLAN.md` are Mermaid, which GitHub renders automatically.

## How to set up a repo

1. Create the repo, copy the three files for that project into the root. For Tells, also copy `docs/`, `scripts/`, and `data/` (Phase 0 output); `data/raw/` must be gitignored on Day 1.
2. Open Claude Code in the repo and paste the kickoff prompt from `PROMPT.md`.
3. Each day, paste the daily prompt with the day number.

## Honest limits

- Tells has results (`tells/docs/findings.md`). Every bracketed number in a pitch script is still a placeholder to fill only with something you measured.
- Dataset names, TSFM capabilities, and the SR 26-2 details come from web sources and must be verified at build time. The plans say where. Tells datasets were measured on 2026-11-10 (`tells/docs/data-fit.md`).
- Culprit uses simulated data only. Say so every time you present it.
