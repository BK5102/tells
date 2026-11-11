# PROMPT.md: Tells

Prompts to paste into Claude Code. Put `CLAUDE.md`, `PROMPT.md`, and `PLAN.md` at the repo root first.

---

## 1. Kickoff prompt (paste once, on day 1)

```
You are my pair-programmer and teacher. I am a data science and economics student who is new to fraud detection and time series foundation models (TSFMs). 

Read CLAUDE.md, PLAN.md, docs/decisions.md, and docs/data-fit.md in full before doing anything. Then:

1. Summarize the project back to me in 5 sentences of plain English with one real-world analogy, so I can check we agree.
2. Do ONLY Day 1 of PLAN.md Part 6: repo scaffold as a uv workspace (root package tells, member packages/cleanroom), ruff, pytest, CI that tests both packages, a .gitignore that excludes data/raw/, artifacts/, and .env, and a README skeleton that embeds Diagram 1 from PLAN.md. Do not start Day 2.
3. Keep docs/decisions.md and docs/data-fit.md as they are. Create docs/tsfm-notes.md as an empty template.
4. Run tests and lint. Show me the output.
5. End with: what you built (3 sentences, plain English, one analogy), what you did not verify, and the exact first command I should run.

Rules: follow every rule in CLAUDE.md. Do not invent datasets, results, or citations. If you are unsure about something, ask me one specific question instead of guessing. No em-dashes, no emojis.
```

## 2. Daily prompt (paste every day, change the number)

```
Today is Day N of the Tells 30-day plan.

1. Read CLAUDE.md and the Day N row in PLAN.md Part 6. Read docs/decisions.md for yesterday's notes.
2. Restate today's task in two sentences and list the files you will touch. Wait for my "go" if the task changes the Cleanroom or the agent's permissions; otherwise proceed.
3. Write the test first (a leak test if it touches features or splits). Then the code.
4. Run pytest and ruff. Fix failures.
5. Append a 3-sentence note to docs/decisions.md: what I built, why it matters, and a real-world analogy.
6. Propose a commit message in the form "type: short summary".
7. Finish with: what changed, what you did NOT verify, and one question that tests whether I understood today's feature.
```

## 3. Leak audit prompt (run at Days 11, 20, 26)

```
Act as a skeptical reviewer whose only job is to find ways this evaluation could be cheating. Read packages/cleanroom, src/tells/features, src/tells/cases.py, and the latest experiment config.

Check, with evidence from the code:
- Does any feature at time t use rows from time t or later?
- Do train and test overlap in time or by account in a way that leaks?
- Is the sealed-envelope lock enforced in code, and can it be bypassed?
- Does the canary leaky feature get caught? Run it.
- Does the time-shuffled control destroy the signal? Run it.
- Are numbers in the generated report all present in the results files?
- Does any feature use an account's earlier fraud labels without a label delay?
- Is AMLWorld truncated before its end-of-data tail, and is the TabFormer test period the one locked in the config?
- Does packages/cleanroom import anything from tells, or contain fraud-specific code?

Report each check as PASS or FAIL with file and line references. Do not fix anything yet. List fixes in priority order.
```

## 4. Explain-it-to-me prompt (run after each week)

```
Quiz me. Ask me, one at a time, five questions about what we built this week, starting with the simplest. After each answer, tell me what I got right, what I got wrong, and give a better 2-sentence version using a real-world analogy. Finish by writing a 60-second spoken explanation of the project as it stands today, using only things that actually exist in the repo.
```

## 5. Recruiter rehearsal prompt (run on Days 20, 28, 30)

```
Play a recruiter at a fintech company, then a hiring manager on a fraud team, then an ML interviewer. Ask me the questions they would really ask about Tells, one at a time. After each of my answers, score it from 1 to 5 for clarity and honesty, point out any claim I cannot back with something in the repo, and suggest a tighter version. Use the "Questions you will probably get" and "What not to say" sections of PLAN.md Part 4 as the base, but add new questions.
```

## 6. Findings write-up prompt (Day 28)

```
Read artifacts/results (all runs). Write docs/findings.md with these sections: question, setup, headline table, what worked, what did not work, what could not be tested, and next steps. Every number must be copied from a results file and tagged with its config hash. Include negative results. Do not use causal language the experiments cannot support. No em-dashes.
```

## 7. When something breaks

```
Do not guess. Reproduce the failure, show me the smallest failing case, explain the cause in plain English, propose the minimal fix, and add a regression test before changing code.
```
