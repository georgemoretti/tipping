# Plan — `tipping`: finding the point of no return in LLM agent runs

Rules: [CLAUDE.md](CLAUDE.md). Pace: as fast as possible, no fixed schedule.

**Core idea.** Take a recorded agent run. For each step *k*, replay steps 1..k−1 exactly, then run the rest live N times, and measure the success rate. This gives a **success curve** along the run. The steepest drop is the **decisive step**, which often differs from the first step where runs diverge.
**Method contribution:** an adaptive scheduler that finds the decisive step at a fraction of the naive cost (steps × N).

**Order.** The three chapters run side by side. Dependencies between them:
- The literature review (A1) should finish before the method block (B9), so the method is positioned correctly.
- Experiments (A4) need blocks B0–B9.
- Launch posts follow the release (B11) and the arXiv submission (A7).

---

## Chapter 1 — Building the project, block by block

Every block has a **goal**, a **deliverable**, and a **"how you test it"** check. Don't move on until the check passes.

### Hardware (RTX 5060 Ti 16 GB)
- Serve with **vLLM**, which batches many parallel resamples well. Use Ollama only as an easy fallback.
- Models: Qwen3-8B (FP8/AWQ) as the main agent model. Try Qwen3-14B (AWQ, 4-bit) if it fits alongside the KV cache. Use one small model as the τ-bench user simulator.
- Blackwell GPUs need recent CUDA (12.8+) and a recent vLLM/PyTorch build. Verify this in B0.

### Blocks
- [ ] **B0 — Setup.** `uv` project (PyPI name `agent-tipping`), `src/agent_tipping/` layout, ruff, pytest, GitHub Actions CI, repo `georgemoretti/tipping`. On the Windows PC: WSL2 + CUDA + vLLM, serving an OpenAI-compatible API that the Mac can reach over the local network.
  *Test:* `pytest` is green on the Mac. A script on the Mac sends 32 parallel requests to vLLM on the Windows PC and prints tokens/s.
- [ ] **B1 — Toy agent + toy environment.** A tiny LangGraph agent (3–4 tools, e.g. a mini shop) with deterministic tools, 10 tasks, and a known success check per task. Some tasks are designed to have a clear decisive step. It's also the test fixture for all later blocks.
  *Test:* run all tasks with the local model and see a mix of passes and failures.
- [ ] **B2 — Data model + Trace store.** Dataclasses for `Run`, `Step` (node, LLM call, tool call, checkpoint id), `Resample`, `Outcome`. Stored in one SQLite file.
  *Test:* write a run, read it back, get an identical object. Open the file in any SQLite viewer.
- [ ] **B3 — Recorder.** Hook into LangGraph (callbacks / `astream_events` + checkpointer) and save every step with its checkpoint.
  *Test:* record a toy run. The stored steps match what the agent printed, and each checkpoint id restores.
- [ ] **B4 — Forker + environment adapters.** Restore state at step *k*, replay the prefix exactly, and run the suffix live N times (async, rate-limited, seeded). Adapters: `replay` (recorded tool outputs), `reexecute` (deterministic tools), `reset_to(k)` hook.
  *Test:* forking at k = last step reproduces the original outcome. Forking at k = 0 matches fresh runs. The prefix is byte-identical across forks.
- [ ] **B5 — Outcome oracle.** An interface for "did it succeed?": a user function, a task grader, or an optional LLM judge.
  *Test:* the toy tasks are graded correctly, using fixed known traces.
- [ ] **B6 — Naive estimator (the baseline).** Uniform N resamples at every step. Beta posterior per step, success curve, decisive step = largest drop with a confidence interval.
  *Test:* on toy tasks with a built-in decisive step, it finds that step.
- [ ] **B7 — Reporters.** Terminal table, a single-file HTML report with the curve and step details, and JSON output.
  *Test:* open the HTML. The curve and decisive step are understandable without explanation.
- [ ] **B8 — CLI.** `tipping record`, `tipping analyze --budget`, `tipping report`.
  *Test:* the whole flow runs from a clean shell, following only the README.
- [ ] **B9 — Adaptive scheduler (the method).** Coarse probes, then bisection on the largest drop, plus sequential stopping (stop when the confidence intervals separate) and a hard budget.
  *Test:* it finds the same decisive step as B6 at a measured fraction of the cost.
- [ ] **B10 — Pytest plugin.** `@pytest.mark.tipping(runs=…, budget=…)`. A failing or flaky test reports its decisive step.
  *Test:* a deliberately flaky toy test shows "decided at step k" in the pytest output.
- [ ] **B11 — Release v0.1.** README (quickstart, screenshot of the HTML report, comparison with related tools), PyPI package, examples folder, MIT licence.
  *Test:* `pip install agent-tipping` in a fresh environment, then the quickstart works in under 5 minutes.

### Backlog (v0.2+, from the feature list)
- Counterfactual patch test: edit a step and re-estimate success.
- Change-aware test selection.
- Splitting the outcome variance by source (LLM sampling vs tools).
- Import failed traces from LangSmith / Langfuse.
- Decisive-step leaderboard across a test suite.
- More frameworks beyond LangGraph.

---

## Chapter 2 — The arXiv article

**Working title:** *Where Is the Outcome Decided? Efficient Estimation of Decisive Steps in Multi-Step LLM Agent Trajectories*

- [ ] **A1 — Literature review.** Replay and testing (Chronicle, agrepl, record/replay plugins), consistency and divergence (2602.11619, 2605.28840, AgentDiff, 2604.27586, 2609.08832), step-level value / credit assignment in RL for agents, statistical evals (error bars, sequential testing). Output: `paper/related_work.md` with verified references. **Gate:** the novelty claim survives, or we adjust the angle.
- [ ] **A2 — Research questions.**
  - RQ1: What do success curves look like, and where are outcomes decided?
  - RQ2: How often does the decisive step coincide with the first-divergence step?
  - RQ3: How much cheaper is adaptive estimation than uniform sampling, at equal accuracy?
  - RQ4: Do interventions at the decisive step help more than at the first-divergence step or a random step?
- [ ] **A3 — Experiment design.** Benchmarks: toy env (ground truth) + τ-bench / τ²-bench (resettable, deterministic tools). Models: 2–3 local models on the 5060 Ti. Baselines: uniform sampling, first-divergence step, per-step stability (as in 2609.08832). Metrics: localisation error, cost (LLM calls / tokens), CI width.
- [ ] **A4 — Pilot, then main experiments.** All runs logged under `experiments/` with configs and run ids.
- [ ] **A5 — Writing.** Intro → method → setup → results → related work → limitations. Figures: success curves, cost vs accuracy, decisive vs divergence step.
- [ ] **A6 — Supervisor review + arXiv endorsement.** Ask the supervisor early whether they'll be co-author and endorser.
- [ ] **A7 — arXiv submission**, categories cs.CL / cs.AI / cs.SE. Code link in the abstract.
- [ ] **A8 — Venue (later).** ACL/EMNLP System Demonstrations, or an agents/evaluation workshop. Check the 2027 deadlines.

Folder: `paper/` (LaTeX, figures, `related_work.md`), `experiments/` (configs, scripts, raw results).

---

## Chapter 3 — LinkedIn

### Profile (before any posts)
- Headline: *ML/NLP Engineer · LLM Agents & Agent Evaluation · PyTorch, LangGraph · PhD researcher*
- About: 4–5 lines on production agents + the evaluation research.
- Featured section: the repo, later the paper.

### Post rules
- 150–250 words, one idea, one image or diagram, a link in the post or first comment.
- Written for engineers who ship agents, not for researchers. Practical lesson first, research second.
- Every number comes from our own experiments (rule 10).

### Posts (in order, each tied to a milestone)
1. The problem: "Your agent failed. Which step actually caused it? Usually not the one you think." *(any time)*
2. What I learned reading papers on agent consistency (a short map of the field). *(after A1)*
3. Divergence ≠ failure: why "the first step where runs differ" is a misleading debug signal. *(after A1)*
4. Building in public: the idea of replaying the past and resampling the future (diagram). *(after B4)*
5. Running agent experiments on a 16 GB consumer GPU: setup and lessons. *(after B0–B4)*
6. First success curve on a real agent (image). *(after B7)*
7. Why debugging agents with a single run lies to you (numbers from the pilot). *(after A4 pilot)*
8. 🚀 **Launch: `agent-tipping` v0.1** (screenshot, quickstart, repo). *(after B11)*
9. Result: how often the decisive step ≠ first-divergence step. *(after A4)*
10. Result: finding the decisive step at X% of the cost. *(after A4)*
11. 🚀 **Paper on arXiv** (the 3 key findings + links). *(after A7)*

Drafts live in `posts/` as markdown. The owner edits and publishes (rule 13).
