# Execution Tracker V2 (start: Mon Nov 2, 2026, review every Sunday)

Goal: remote, USD-paying AI engineering work (contract, part-time, freelance).
Rule: every claim on LinkedIn, Upwork, or a CV must link to a repo, a live demo, or a committed report within one click.
Source of truth for scope: VISION_AGENT_PRD_V1.md

## Corrections to earlier tracker entries (honest log)

- Removed the "$0.0008 per detection" figure from the V2 draft in chat. No measurement backs it. Real cost comes from the analytics endpoint after M4.
- Earlier PRD template stated "Render free tier: 100 hours/month". Not verified. Do not use it. Current Render plan data is in PRD Section 8.
- Earlier chat recommended YouTube links I had not verified. Treat them as unverified until checked.
- Earlier plan said "deploy on Render free tier". Live demos need a paid plan for an always-on service and a persistent disk for SQLite. Decision is in PRD Section 8.

## Daily rules (weekdays)

- One fixed 90-minute block, same time every day, phone away.
- Split: 60 min build, 15 min outreach plus 5 comments on target people, 15 min log.
- The first step of every day is under 15 minutes and is written the night before.
- End-of-block report to Claude, in this format:
  `Done: ... | Evidence: <link, commit or file> | Blocked: ... | Tomorrow's first 15 min: ...`
- Missed a day: log "missed, reason" and do the smallest version next day. Never skip two days in a row.
- Deploy budget: 8 deploys maximum for Project 2. Log each deploy with its reason in the deploy log below.

## Phase 0: Pre-flight (Mon Nov 2) — must finish before any code

| #    | Task                                                                           | Est. | Done | Evidence |
| ---- | ------------------------------------------------------------------------------ | ---- | ---- | -------- |
| 0.1  | PRD section 13 accounts: Render account and payment method                     | 15m  | []   |          |
| 0.2  | Decide frontend host: Vercel Pro ($20) or commercial-use static host ($0)      | 10m  | [ ]  |          |
| 0.3  | Groq API key created and stored outside git                                    | 10m  | [ ]  | Done     |
| 0.4  | Confirm Groq model supports tool calling (3 test prompts)                      | 20m  | [ ]  | Done     |
| 0.5  | Python 3.11 and Docker Desktop verified                                        | 15m  | [ ]  | Done     |
| 0.6  | Git repo `vision-agent` created, `.gitignore` and `.env.example` written | 20m  | [ ]  |          |
| 0.7  | 6 test images chosen with source/licence noted                                 | 20m  | [ ]  |          |
| 0.8  | 20 eval questions drafted (before agent is built)                              | 30m  | [ ]  |          |
| 0.9  | Monthly budget written in PRD section 13                                       | 5m   | [ ]  |          |
| 0.10 | LinkedIn "Open to work" set to recruiters; CiteCraft pinned on profile         | 15m  | [ ]  |          |

Phase 0 exit test: all rows ticked with evidence, PRD signed off in writing.

## Milestone plan (from PRD section 14)

| Milestone           | Days                     | Done when                                                                         | Done    | Evidence |
| ------------------- | ------------------------ | --------------------------------------------------------------------------------- | ------- | -------- |
| M1 Model            | Tue Nov 3 – Wed Nov 4   | yolov8n ONNX exported, benchmark script prints p50/p95, RSS measured, plan chosen | [ Done] |          |
| M2 Service          | Thu Nov 5 – Fri Nov 6   | `/detect`, `/health`, schema, guards, auth, pytest green                      | [ ]     |          |
| M3 Agent            | Sat Nov 7 – Sun Nov 8   | loop, 3 tools, `/ask`, mocked-LLM tests cover all PRD section 9 failure rows    | [ ]     |          |
| M4 Eval + analytics | Mon Nov 9                | 20-case eval run, `eval/report_v1.md`, `/analytics`, injection tests          | [ ]     |          |
| M5 Deploy backend   | Tue Nov 10               | Docker builds, live on Render, one real `/ask` via public URL                   | [ ]     |          |
| M6 Frontend         | Wed Nov 11               | upload, boxes, ask, tool trace; deployed                                          | [ ]     |          |
| M7 Proof            | Thu Nov 12 – Fri Nov 13 | README with measured numbers, 45-sec recording, post drafted                      | [ ]     |          |
| M8 Publish          | Sat Nov 14               | post live, 5 DMs, 5 comments, 10 applications                                     | [ ]     |          |

## Week 1 day-by-day (Nov 2 – Nov 8)

| Day   | Build task (60 min)                                                               | First 15-min step (write the night before)                     | Done | Evidence |
| ----- | --------------------------------------------------------------------------------- | -------------------------------------------------------------- | ---- | -------- |
| Mon 2 | Phase 0 pre-flight (replaces build block today)                                   | Open Render and Groq dashboards                                | [ ]  |          |
| Tue 3 | Export yolov8n to ONNX; run one image through onnxruntime                         | `pip install ultralytics onnxruntime pillow` in a clean venv | [ ]  |          |
| Wed 4 | Benchmark script: 100 runs, p50/p95, RSS via psutil; plan decision                | Write `bench/benchmark.py` skeleton                          | [ ]  |          |
| Thu 5 | `detector.py` + `/detect` route + `/health`                                 | Create `app/` package and empty `main.py`                  | [ ]  |          |
| Fri 6 | SQLite schema (`db.py`), guards (magic bytes, size), API-key dependency, pytest | Write `schema.sql` from PRD section 6                        | [ ]  |          |
| Sat 7 | Agent loop (`agent.py`) hand-written, with mocked LLM                           | Write the loop as pseudocode comments first                    | [ ]  |          |
| Sun 8 | Three tools with pydantic args; tool-result wrapping                              | Write the `count_labels` argument model                      | [ ]  |          |

Sunday 8 review (15 min): fill the weekly table below.

## Week 2 (Nov 9 – Nov 15)

- Mon 9: eval harness, 20 cases, `report_v1.md`, analytics endpoint, injection tests
- Tue 10: Dockerfile, deploy to Render, verify `/health` and one `/ask` on public URL
- Wed 11: frontend (reuse CiteCraft layout patterns), deploy
- Thu 12 – Fri 13: README numbers from analytics endpoint; 45-sec screen recording
- Sat 14: publish post and outreach block

## Deploy log (max 8)

| # | Date | Target | Reason | Result |
| - | ---- | ------ | ------ | ------ |
| 1 |      |        |        |        |

## Outreach quota (from Nov 9)

- 10 applications or proposals per week
- 5 DMs per week, personalised, each linking a live URL
- 5 comments per weekday on target people (recruiters, hiring managers, AI engineers)

## Weekly review (every Sunday, 15 min)

| Week                         | Applications sent | DMs | Posts | Profile views | Search appearances | Replies / interviews | Commits | Notes                        |
| ---------------------------- | ----------------- | --- | ----- | ------------- | ------------------ | -------------------- | ------- | ---------------------------- |
| Baseline (Oct 25, CiteCraft) | ~15               | ~8  | 1     | TBD           | TBD                | 1 (Andela)           | 25+     | 84 impressions on first post |
| Wk 1 (Nov 2–8)              |                   |     |       |               |                    |                      |         |                              |
| Wk 2 (Nov 9–15)             |                   |     |       |               |                    |                      |         |                              |
| Wk 3                         |                   |     |       |               |                    |                      |         |                              |
| Wk 4                         |                   |     |       |               |                    |                      |         |                              |

## Daily ledger

| Date      | Planned   | Done? | Evidence | Blocked by | Next 15-min step |
| --------- | --------- | ----- | -------- | ---------- | ---------------- |
| Mon Nov 2 | Phase 0   |       |          |            |                  |
| Tue Nov 3 | M1 export |       |          |            |                  |

## Is this on track? (review at each Sunday)

- Green: M-milestone done on or before its target day, deploy count within budget, evidence linked.
- Amber: one milestone slipped by 1 day, reason logged.
- Red: two milestones slipped, or a deploy was made without a logged reason. Stop and re-plan before the next build day.
