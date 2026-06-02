# SayCan-Gesture: Working Notes for Claude Code

This file is the operating manual for working in this repo. The full system
specification is in **SPEC.md** — read it first. This document covers *how*
to work here; SPEC.md covers *what* to build.

---

## Project in one paragraph

We're building a co-speech gesture selection system for the Pepper robot that
extends the SayCan factorization (LLM × affordance) with a third online-learned
term ("Adapt") plus closed-loop feedback. The headline scientific claim is that
the Adapt term and feedback loop produce measurable improvement over the
vanilla-SayCan port across simulated long-term interactions. Target: an HRI
conference paper. The codebase has to support a clean, fair comparison against
that vanilla SayCan baseline — that comparison is the central result and
nothing must contaminate it.

---

## Read this before writing any code

1. **SPEC.md** — full system spec, section by section.
2. **experiments/ablation_matrix.md** — what configurations exist and why.
3. **experiments/metrics.md** — how everything gets measured.
4. **src/pipeline/selector.py** — the core scoring rule, already scaffolded.
5. **tests/test_selector_identity.py** — the M0 ≡ B1 guardrail; understand it.

If you're about to make a design decision that isn't covered in SPEC.md, stop
and either ask the human or document it as a `# ASSUMPTION:` in code AND in
`BUILD_NOTES.md`. Silent assumptions are the failure mode that kills research
code.

---

## Build order — follow it strictly

SPEC.md §10 defines a strict build order. Do not skip ahead, even if a later
component feels "easier" or "more interesting." The order is designed so each
layer has its tests passing before the next is built on top of it. The
sequence:

1. Foundation (`pyproject.toml`, `RobotInterface`, `SimPlayback`, library loader)
2. Data layer (loaders, splits, simulated users)
3. Can scorer (simplest scorer; sets the pattern)
4. Say scorer (LLM client + prompt + cache)
5. Adapt scorers (three variants: bandit → memory → adapter)
6. Feedback sources
7. Pipeline / orchestrator / logging
8. Baselines (B1 first — it's the identity check)
9. Eval (metrics, runner, leaderboard; LLM judge last)

After each step, run the tests for what you just built. Don't proceed if they
fail. Don't proceed if you skipped writing them.

---

## Non-negotiables

These are the rules that protect the scientific integrity of the result. Do
not violate them under any circumstance, even if a refactor would be cleaner.

### 1. The M0 ≡ B1 identity must hold
Our method with `adapt=None` and `weights.d=0` must produce numerically
identical selections to the vanilla SayCan baseline (B1). There is a unit test
enforcing this (`tests/test_selector_identity.py`). If you change the selector,
this test must still pass. If it ever fails, **stop everything** and figure out
which side drifted before doing anything else. A drift here invalidates every
"improvement over SayCan" claim downstream.

### 2. Baselines are implemented faithfully
B2 (Wake 2023) and B3 (Torshizi 2025) are reimplemented as described in their
papers. They do NOT get our improvements bolted on, even partially. They do
NOT get nerfed to flatter our method. They share our gesture library,
`RobotInterface`, data loaders — everything except the selection logic. Where
a paper is underspecified, document the assumption with `# ASSUMPTION:` in
code and a note in `BUILD_NOTES.md`. If unsure whether something counts as
faithful, ask.

### 3. All randomness is seeded
Every run is fully reproducible from `(config_yaml, seed, git_commit)`. All
randomness flows through a single `numpy.random.Generator` instantiated from
the seed in the config. No bare `random.random()`. No `np.random.*` without a
generator. This is non-negotiable for reproducibility.

### 4. No fabricated data, numbers, or results
If a metric hasn't been computed, say so. If a dataset count is unknown, look
it up rather than estimate. If a run hasn't happened, don't fill the
leaderboard with placeholder rows. The paper depends on every reported number
tracing to a real run.

### 5. Cold start in Adapt scorers
All three Adapt variants must return uniform-or-near-uniform log-probabilities
when a user has zero history. This ensures day-1 behavior matches vanilla
SayCan; adaptation only kicks in once feedback accumulates. There is a unit
test for this. Don't relax it.

### 6. Speaker-disjoint splits
Train / val / test splits never share a speaker. There is a unit test
asserting this. Don't relax it.

### 7. The LLM cache is mandatory
Every Say scorer call goes through the disk cache. Re-running the same config
with the same cache must produce byte-identical results. Without this,
experiments aren't reproducible and the API bill becomes unbounded.

---

## Coding standards

- **Python 3.11+** for everything except the Pepper bridge (which is Python 2.7
  because NAOqi requires it; it runs as a subprocess and talks JSON-RPC).
- **Type hints everywhere.** `mypy --strict` must pass on `src/`.
- **`ruff` for lint and format.** Run before committing.
- **`pytest` for tests.** Every scorer, every metric, every loader gets at
  least one test. Test files go in `tests/` and mirror the `src/` layout.
- **No bare `except`.** Catch what you mean to catch.
- **No `print` in `src/`.** Use the project logger from `src/pipeline/logging.py`.
- **Configs are typed.** Load YAML into dataclasses (`pydantic` or `attrs`),
  not into raw dicts. This is what catches typo'd config keys at load time.
- **Protocols over inheritance.** Scorers, robots, feedback sources are
  `typing.Protocol` types. Don't introduce abstract base classes.
- **Pure functions where possible.** Especially for metrics and checks. Easier
  to test, easier to reason about.

---

## Repo conventions

### Directory layout
The target layout is in SPEC.md §10. Follow it exactly. Don't invent new
top-level directories without asking.

### Where things live
- Prompts go in `src/<module>/prompts/<version>.txt` and are loaded as
  files — never inline a prompt in Python code. Versioning prompts is how we
  enable prompt ablations.
- Configs go in `configs/`. One file per experiment. Shared defaults live in
  `configs/base.yaml` and individual configs override.
- Data goes in `data/`. Raw datasets are git-ignored (too big); the library
  YAML and the dataset-to-library mapping are committed.
- Results go in `results/<run_id>/`. The leaderboard is
  `results/leaderboard.json` and is append-only — never edit historical rows.
- Tests mirror `src/`: `src/can/scorer.py` → `tests/test_can_scorer.py`.

### Imports
- Use absolute imports (`from src.can.scorer import CanScorer`), not relative.
- Group: stdlib → third-party → project, with blank lines between groups.
- Don't `import *`. Ever.

### Git hygiene
- Small commits, one logical change each.
- Commit messages: `<area>: <what>` (e.g. `can: add joint-range check`).
- Don't commit large data files or model weights. Don't commit `.env` or
  anything in `data/cache/`.

---

## Logging and observability

Use the project logger (`src/pipeline/logging.py`), which writes:
- Console output at INFO by default.
- Structured per-turn rows to `results/<run_id>/per_turn.parquet`.
- A `run.log` file in each run directory.

Each per-turn row records: turn index, user_id, utterance, chosen gesture,
the four scores (say, can, adapt, repetition), the combined score, reward,
latency, and any fallback flags (`say_fallback=True` if the LLM call failed).
These rows are how all metrics get computed and how the paper's figures
get made. Don't skip fields; downstream code expects them.

---

## When you're stuck or uncertain

- If SPEC.md is silent on something, prefer the choice that makes scientific
  integrity easier (simpler, more transparent, more testable) over the choice
  that makes the headline number bigger.
- If two designs are equally valid, pick the one with fewer moving parts.
- If you've spent more than ~15 minutes guessing at what the spec means, stop
  and ask. The human would rather answer one question now than untangle a
  wrong assumption later.
- If a test you wrote is hard to make pass, the test is usually right and the
  code is usually wrong. Don't weaken the test to make it green.

---

## What "done" looks like for v1

SPEC.md §13 has the full checklist. Summarized:

1. All tests pass (`pytest tests/ -v`), especially `test_selector_identity.py`
   and `test_baselines_parity.py`.
2. `mypy --strict src/` is clean.
3. `ruff check src/ tests/` is clean.
4. Every config in `configs/` (B1–B4, M0–M3, plus ablations) runs end-to-end
   via `python -m src.eval.runner --config <path>` and writes a leaderboard
   row.
5. `BUILD_NOTES.md` exists at the root, documenting every `# ASSUMPTION:` made
   during the build.

When these are all green, stop and report back. We'll add the experimental
sweep orchestration as a separate phase — not as part of this build.

---

## What NOT to do during the v1 build

- **Don't implement gesture generation.** Stub the interface; do not build it.
  Continuous motion synthesis is explicitly out of scope.
- **Don't implement vision-based feedback.** The `FeedbackSource` interface
  allows it; only `ExplicitRatingFeedback` and `SimulatedEngagementFeedback`
  ship in v1.
- **Don't add new Adapt variants.** Three is the right number for the paper.
  More variants = harder to compare = weaker contribution claim.
- **Don't optimize for real-time latency.** Measure it, report it, but don't
  contort the architecture for it. Pepper is slow; we have headroom.
- **Don't build the Claude Code agent orchestration.** That's a separate phase
  after v1 is green. Focus is on a clean, tested, reproducible codebase.
- **Don't refactor `tests/test_selector_identity.py`** to make it weaker. If
  it's failing, fix the code, not the test.

---

## A note on honesty

This is research code that will produce numbers in a conference paper. The
single most important thing is that the numbers are real and the comparison
is fair. It's much better to ship a v1 with one honest result than a v1 with
several flattering results that don't survive scrutiny. If you find yourself
considering a shortcut that would make our method look better against the
baselines, that's the moment to stop and surface it instead.