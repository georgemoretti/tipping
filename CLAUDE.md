# Project rules — `tipping`

`tipping` finds the **point of no return** in an LLM agent run: the step where success or failure was decided.
It replays a recorded run up to step *k*, resamples the rest live N times, and estimates P(success | prefix) per step.
Full plan: [PLAN.md](PLAN.md).

## Context
- Owner: ML/NLP engineer (4 yrs, LLM agents in production), 1st-year PhD on *Methods for Reliable and Efficient Evaluation of Multi-Step LLM Agents*.
- Goal: open-source tool + arXiv paper + LinkedIn posts → ML/AI engineer offer in the Netherlands within ~6 months (from Nov 2026).
- Hardware: RTX 5060 Ti 16 GB on a **separate Windows PC** (vLLM runs in WSL2). Code is written on the Mac. Stack: Python, PyTorch, LangChain/LangGraph.
- Code ownership: **Claude writes each block with tests; the owner runs the checks and confirms.**
- Repo: `github.com/georgemoretti/tipping` (public). PyPI package: **`agent-tipping`**, import name `agent_tipping`, CLI command `tipping`. (`tipping` is taken on PyPI.)

## Rules Claude follows

### Working style
0. **Never add anything on my own.** No new tasks, sections, tracks, features or schedules in the plan, code or docs unless the owner asked. Propose the idea and ask first. No week-by-week timelines: the owner works as fast as possible.
1. **One block at a time.** Work only on the current block from PLAN.md. Each block ends runnable and tested. The owner runs it and confirms before the next block starts.
2. **Explain before code.** For each block, first give a short design note: what it does, its interface, and why. Then write the code. Then say exactly how the owner can test it.
3. **Small and readable.** The core should be readable in an afternoon. Prefer plain functions and dataclasses. Add a dependency only with a stated reason.
4. **Tests ship with code.** Every block adds pytest tests, which run offline with a fake LLM. Tests that need a GPU or network are marked and skipped by default.
5. **Scope guard.** Ideas beyond the current block are proposed to the owner. Only those the owner approves go to the *Backlog* section in PLAN.md. None go straight into the code.
6. **Update PLAN.md** status checkboxes when a block is confirmed done.

### Cost and compute
7. **Local first.** By default, use local models on the 5060 Ti. Never call a paid API, or start a long GPU job (>30 min), without asking first.
8. Every resampling job takes a budget (runs / $ / time) and stops when the budget is used up.

### Honesty and science
9. **No unsupported novelty claims.** Every "first / nobody has done" claim needs a literature check, with sources listed in `paper/related_work.md`.
10. **No fabricated citations or numbers.** Verify every reference exists. Every number in the paper or a post must trace back to a script, config and run id in `experiments/`.
11. **Reproducibility.** Fix seeds where possible, log model and version, save raw results, and keep configs in git.
12. **Report negative results** honestly. If the method doesn't beat a baseline, say so, then adjust.

### Posts and the public
13. Posts are drafts in the owner's voice. The owner edits and publishes. Claude never posts or sends anything.
14. Text posts and images/diagrams only; no videos.
15. Never use the employer's confidential information, code or data.

### Git
16. Commit only when the owner asks. Use clear messages, one block per commit where possible.
17. **No Claude attribution.** Never add `Co-Authored-By: Claude` or any "Generated with Claude Code" line to commits or PRs. The owner is the only contributor.
