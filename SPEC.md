# SayCan-Gesture: System Specification

> A complete build brief for an LLM-and-feasibility-grounded co-speech gesture
> selection system for the Pepper robot, with online user adaptation. Hand this
> document to Claude Code as the source of truth.

---

## 1. Project overview

### What this system does
Given a robot utterance and conversational context, pick the **best gesture** for the
robot to perform from a fixed library of discrete gestures, and learn over time which
gestures work best for each specific user.

### Why this exists
Existing LLM-based gesture selection methods (Wake et al. 2023; Torshizi et al. 2025)
ask a language model to pick gestures from a library. They work, but they have three
gaps:

1. They don't check whether the gesture is **physically feasible** on the robot right
   now (current pose, joint limits, time budget before the next utterance).
2. They don't **adapt** to specific users over time. Day 1 and day 30 produce the same
   selections regardless of how the user reacted.
3. They are **open-loop**. Once a gesture is selected, the system never learns from
   how it was received.

The SayCan paper (Ahn et al. 2022) introduced a clean factorization for grounded
action selection — LLM relevance multiplied by a learned affordance — but it was
applied to kitchen manipulation, not gesture, and it is also static and open-loop.

### What we contribute
A three-term selection rule that extends SayCan to co-speech gesture:

```
score(g) = a·log p_say(g | utterance, context)        ← LLM semantic relevance
         + b·log p_can(g | robot_state, time_budget)  ← embodiment feasibility
         + c·log p_adapt(g | user_model, history)     ← online learned appropriateness
         − d·repetition_penalty(g, recent_gestures)   ← long-horizon regularizer
```

Plus a closed feedback loop: after each gesture, an engagement signal updates the
Adapt term so the system gets better at each user over time.

Target venue: an HRI or multimodal-agents conference. The paper's central claim is
that the Adapt term and feedback loop produce measurable improvement over the
vanilla-SayCan port (Say × Can only) across simulated long-term interactions.

### Honest scope for v1
- **Discrete gesture library selection only.** Generation is future work.
- **Pepper as the target robot.** Other embodiments are new adapters, not a rewrite.
- **Offline evaluation + simulated long-term protocol.** Live human study is a
  separate, later effort.
- The `Can` term in v1 is **rule-based** (joint limits, pose validity, duration vs.
  time budget). A learned naturalness score is a v2 extension.

---

## 2. Problem formulation

### Inputs at every decision step
- `utterance: str` — the text the robot is about to say (or is saying).
- `dialogue_history: list[str]` — previous turns, both robot and user.
- `robot_state: RobotState` — current pose, joint positions, time-since-last-gesture,
  time-budget-until-next-speech-boundary.
- `user_id: str` — which user is being interacted with (for the Adapt term).
- `feedback_history: list[FeedbackEvent]` — accumulated reactions from this user
  across all prior sessions.

### Output
A single `Gesture` from the library `G`, plus a record of the scores that produced
the selection (for analysis and debugging).

### The selection rule
For each candidate `g ∈ G`, compute three log-probabilities and a penalty, sum them
with tunable weights, and take the argmax:

```
chosen = argmax_{g ∈ G} [
    a · log p_say(g | utterance, dialogue_history)
  + b · log p_can(g | robot_state, time_budget)
  + c · log p_adapt(g | user_id, feedback_history)
  − d · repetition_penalty(g, recent_gestures)
]
```

`(a, b, c, d)` are weights. Setting `c = 0, d = 0` reduces this to the vanilla SayCan
formulation, which is BOTH a baseline AND the central ablation. The math has to
reduce *exactly* — there is a unit test enforcing this.

### Closed-loop update
After execution, a reward signal `r ∈ [0, 1]` is derived from a feedback source (in
v1: explicit rating or simulated engagement). This calls:

```
adapt.update(gesture, utterance, user_id, reward)
```

Which updates the Adapt scorer's internal state. The next selection step uses the
updated state. This is what makes the system long-term-adaptive.

---

## 3. System architecture

### Components and data flow

```
   utterance ──┐
   history ────┼──→ [Say scorer]   ──┐
                                      │
   robot_state ──→ [Can scorer]   ──┤
                                      ├──→ [Combiner] ──→ argmax ──→ [RobotInterface.execute]
   user_id ─────→ [Adapt scorer] ──┤                                            │
   feedback_history                  │                                            │
                                      │                                            ▼
   recent_gestures ─→ [Rep. penalty]─┘                                  [Feedback collector]
                                                                                    │
                                                                                    ▼
                                                                  reward signal → Adapt.update()
```

### Key abstractions (all are Python `Protocol` types)

- `SayScorer` — takes utterance + history + candidate list, returns log-prob per
  gesture.
- `CanScorer` — takes robot state + candidate list, returns log-prob per gesture.
- `AdaptScorer` — takes user + history + candidates, returns log-prob per gesture;
  also has an `update()` method for closed-loop learning.
- `RobotInterface` — owns the gesture library, exposes state, executes gestures.
  Adapters: `PepperAdapter` (NAOqi), `SimPlayback` (deterministic stand-in for tests
  and offline eval).
- `Selector` — composes the four terms into a selection rule. The only piece of code
  that knows about all three scorers.
- `FeedbackSource` — produces a reward in `[0, 1]` after a gesture executes.
  v1 implementations: `ExplicitRatingFeedback`, `SimulatedEngagementFeedback`.
- `Pipeline` — the orchestrator that runs one interaction turn end-to-end and writes
  a structured log row.

The contract is: **everything talks through these protocols.** Method code never
imports Pepper-specific or LLM-vendor-specific code directly.

---

## 4. The three scorers in detail

### 4.1 Say — LLM semantic relevance

**Question answered:** Does this gesture's meaning fit what the robot is saying?

**Implementation:** Scoring-mode LLM call. We do NOT ask the LLM to generate; we ask
it to score each candidate. This is the same trick SayCan uses — it gives explicit,
calibrated probabilities over the discrete library.

**Inputs:** utterance, recent dialogue history (last N turns, configurable; default
N=6), the full gesture library with one-sentence descriptions and tags.

**Output:** `dict[gesture_id, float]` where values are log-probabilities. Higher =
more semantically appropriate.

**v1 backend:** an API call to Claude (Anthropic API) with a structured prompt that:
- Lists each gesture by `id`, `label`, `description`, `tags`.
- Provides the utterance and history.
- Asks for a score in `[0, 1]` for each gesture (returns JSON).
- We take `log(score + epsilon)` to convert to log-probability.

**Implementation requirements:**
- The prompt is a versioned file: `src/say/prompts/v1.txt`. Never inline a prompt
  in code; version it so we can ablate prompts.
- Cache identical (utterance, history, library_hash) calls to disk to avoid burning
  API budget during reruns. Cache is a simple SQLite file at
  `data/cache/say_cache.sqlite`.
- Vendor abstraction: `LLMClient` protocol with one method `score_gestures()`.
  v1 has `AnthropicLLMClient`. A future `OpenAILLMClient` should drop in without
  touching the Say scorer.
- Failure mode: if the LLM call fails or returns malformed JSON, log the error,
  return uniform log-probability over all candidates, and mark the row in the
  output log as `say_fallback=True`. Never crash the pipeline.

**Files:**
- `src/say/scorer.py` — the `SayScorer` implementation.
- `src/say/llm_client.py` — `LLMClient` protocol and `AnthropicLLMClient`.
- `src/say/prompts/v1.txt` — the scoring prompt.
- `src/say/cache.py` — disk cache layer.

### 4.2 Can — embodiment feasibility

**Question answered:** Can Pepper physically execute this gesture from its current
state, within the available time?

**Implementation in v1:** rule-based, three checks combined:

1. **Pose compatibility.** Each gesture declares `requires_pose` (e.g. "Stand"). If
   the current robot pose doesn't match, score is `log(0.01)` (effectively zero).
2. **Joint range.** Each gesture declares `joints_used`. Check that current joint
   positions are within the gesture's required starting range. If not, score
   degrades linearly with how far out of range we are.
3. **Time budget.** Each gesture declares a nominal `duration_s`. If
   `time_budget_s < duration_s`, score degrades; if budget is comfortably larger,
   full score.

The three sub-scores are multiplied (summed in log space).

**Inputs:** robot state, gesture list.

**Output:** `dict[gesture_id, float]` of log-probabilities.

**Design decision:** v1 is intentionally rule-based, not a learned value function.
SayCan used a learned VF because manipulation success is hard to predict. For
Pepper gesture, the failure modes are well-defined and rule-based feasibility is
more honest and easier to defend in a paper. A learned naturalness scorer is a
clearly-labeled v2 extension.

**Files:**
- `src/can/scorer.py` — the `CanScorer` implementation.
- `src/can/checks.py` — the three sub-checks as pure functions, each unit tested.

### 4.3 Adapt — online user model

**Question answered:** Given what we've learned about *this user* across all prior
interactions, how appropriate is this gesture for them right now?

**Implementation:** Three candidate variants live in the codebase from day one,
sharing the same `AdaptScorer` interface. The paper compares them empirically.

#### Variant A — `BanditAdaptScorer`
Contextual bandit. Each (user, context-bucket, gesture) tuple has a Beta(α, β)
posterior over reward. Scoring uses Thompson sampling: draw a sample from each
posterior; the sample is the score. Update: increment α on positive reward, β on
negative. The "context bucket" in v1 is a coarse label of the utterance
(greeting / question / statement / farewell / other), derived by a simple
keyword/length heuristic — kept deliberately simple so the bandit has enough data
per bucket to learn from.

Why this variant exists: it's the most faithful extension of SayCan's
"value-function-as-affordance" intuition. The Adapt term *is* a learned online
value, just measuring appropriateness instead of feasibility.

#### Variant B — `MemoryAdaptScorer` (RAG-style)
Stores every (utterance, gesture, reward) tuple per user. At scoring time, retrieves
the K most semantically similar past utterances (cosine similarity over sentence
embeddings) and computes, for each candidate gesture, an average reward weighted by
similarity. No training; pure retrieval.

Why this variant exists: it's the cheapest possible adaptation mechanism and a
strong baseline. If a complex method doesn't beat memory-based retrieval, the
complexity isn't justified.

#### Variant C — `AdapterAdaptScorer`
A small per-user linear adapter on top of a frozen sentence embedding of the
utterance. Inputs: utterance embedding concatenated with gesture embedding (a
learned 32-dim embedding per gesture). Output: scalar score. Trained online via
gradient updates on a small in-memory buffer of recent (utterance, gesture, reward)
samples for this user. Population-level priors used until enough per-user data
accumulates.

Why this variant exists: it can generalize to unseen utterance/gesture combinations
in a way pure memory can't, at the cost of more moving parts.

**Common interface for all three:**
- `score(gestures, utterance, dialogue_history, user_id) → dict[gesture_id, float]`
- `update(gesture, utterance, user_id, reward) → None`
- `state_dict() → dict` for checkpointing per user.
- `load_state_dict(d)` for resuming across sessions.

**Cold-start behavior:** with zero history for a user, every variant must return
a uniform-or-near-uniform log-prob distribution. This means on day 1, the Adapt
term contributes nothing and the system behaves like vanilla SayCan. Adaptation
only kicks in once feedback accumulates. This is a hard requirement and there is
a unit test for it.

**Files:**
- `src/adapt/base.py` — the `AdaptScorer` protocol.
- `src/adapt/bandit.py` — Variant A.
- `src/adapt/memory.py` — Variant B.
- `src/adapt/adapter.py` — Variant C.
- `src/adapt/context_bucket.py` — utterance-to-bucket mapping for the bandit.
- `src/adapt/embeddings.py` — sentence embedding helper (uses a small local model;
  `sentence-transformers/all-MiniLM-L6-v2` is the v1 default).

### 4.4 Repetition penalty

**Question answered:** Has this gesture been used a lot in the recent past?

**Implementation:** Maintain a rolling window of the last K selected gestures
(default K=5). Penalty for gesture `g` is `count(g in recent_window) × d` where
`d` is the weight from `SelectorWeights`.

This lives inside `Selector`, not as a separate scorer protocol — it has no
external dependencies and no learning component.

---

## 5. Closed-loop feedback

### Why
Static gesture selection produces robots that don't get better. The whole point of
the Adapt term is that it gets updated, and that requires a reward signal.

### Reward sources

#### `ExplicitRatingFeedback`
The simplest source. After a gesture executes, the experimenter (or the user, in a
study) provides a rating in `[0, 1]`. This is the gold-standard signal for offline
eval where humans label the dataset. It's also what a live HRI study would use with
post-trial Likert ratings, normalized.

#### `SimulatedEngagementFeedback`
For the simulated long-term protocol. Given a labeled dataset of
(utterance, ground-truth-gesture) pairs, the reward for selecting `g` for utterance
`u` is:
- `1.0` if `g == ground_truth(u)`,
- `0.5` if `g` shares any tag with `ground_truth(u)` (partial credit),
- `0.0` otherwise.

This is a proxy. The paper must clearly mark every result from this source as
"simulated" and not represent it as human feedback.

#### `FeedbackSource` protocol
```python
class FeedbackSource(Protocol):
    def get_reward(self, gesture: Gesture, utterance: str,
                   ground_truth: GroundTruthInfo | None,
                   user_id: str) -> float: ...
```

**Files:**
- `src/pipeline/feedback.py` — protocol and both implementations.

### Where the loop closes
In `Pipeline.run_turn()`:

```python
chosen = selector.select(...)
result = robot.execute(chosen)
reward = feedback_source.get_reward(chosen, utterance, ground_truth, user_id)
selector.observe_reward(chosen, utterance, user_id, reward)
log_writer.write_row({...})
```

`selector.observe_reward` delegates to `adapt.update` if Adapt is configured;
otherwise it's a no-op. This means turning the loop off (for the vanilla SayCan
baseline) is just `adapt=None`.

---

## 6. Gesture library and robot interface

### 6.1 The `Gesture` dataclass
Already defined in `src/robot/interface.py` (see existing scaffold). Frozen
dataclass with: `gesture_id`, `label`, `description`, `duration_s`, `requires_pose`,
`joints_used`, `tags`.

### 6.2 The Pepper gesture library
A YAML file at `data/library/pepper_v1.yaml` listing all gestures. Each entry maps
to a NAOqi animation (e.g. `animations/Stand/Gestures/Hey_1`) and includes its
metadata. v1 starts with a curated set of ~30 gestures covering common categories:

- Greetings (wave variants)
- Deictic / pointing
- Beats (small rhythm gestures)
- Iconic (representational: "big", "small", "this much")
- Affect (nod, shake, shrug)
- Discourse markers ("on the one hand...", counting)
- Rest / idle

The library is data, not code. A loader (`src/robot/library_loader.py`) parses the
YAML into a list of `Gesture` objects. Adding a gesture = editing the YAML.

### 6.3 `RobotInterface` adapters

#### `SimPlayback`
The workhorse for development and offline eval. Holds the library, exposes a
configurable `RobotState`, and `execute()` returns a deterministic `ExecutionResult`
based on simple feasibility rules. No real robot, no networking, runs everywhere.

#### `PepperAdapter`
Wraps the NAOqi Python SDK. v1 requirement: **must run as a separate process**
because NAOqi's mature SDK is Python 2.7-bound. The main pipeline (Python 3) talks
to it over a small JSON-RPC layer. Concretely:

- `src/robot/pepper/adapter.py` — Python 3 client; implements `RobotInterface`.
- `src/robot/pepper/server.py` — Python 2.7 server that imports NAOqi.
- Communication: localhost HTTP, simple request/response.

**Hardware availability is not a precondition for any other work.** Everything
above `RobotInterface` runs against `SimPlayback`. The Pepper adapter is built and
tested in isolation, then plugged in for hardware validation runs.

**Files:**
- `src/robot/interface.py` (exists)
- `src/robot/sim_playback.py`
- `src/robot/library_loader.py`
- `src/robot/pepper/adapter.py`
- `src/robot/pepper/server.py` (Python 2.7)

---

## 7. Data

### Datasets in scope for v1
- **BEAT / BEAT2** — large gesture dataset with text and emotion labels. Primary
  source for utterance/gesture pairs.
- **TED-Gesture** — speech-gesture pairs from TED talks. Useful for diversity.
- **(Optional) ZEGGS** — stylistic variation; nice-to-have, not required.

Every dataset gesture must be mapped to a Pepper library entry (or marked
unmapped and excluded). The mapping is a hand-curated YAML at
`data/library/dataset_to_pepper.yaml`. Building this is its own task and the
paper's appendix must show it.

### Splits
Standard train / val / test by **speaker identity** — no speaker appears in
multiple splits. There is a unit test asserting this.

### Simulated user streams (for the long-term protocol)
Generate N synthetic "users" (default 20), each with:
- A preference profile: a probability distribution over gesture tags expressing
  what this user "likes" (e.g. user 3 likes nods, dislikes big arm waves).
- A sequence of M utterances drawn from the dataset (default M=50 per session,
  10 sessions per user).

For each (user, utterance) pair, `SimulatedEngagementFeedback` consults both the
ground-truth gesture *and* the user's preference profile to derive the reward.
This is what creates a learnable per-user signal — without it, every user would
look the same and the Adapt term couldn't beat the Say-only baseline.

**Files:**
- `src/data/loaders.py` — dataset loaders.
- `src/data/splits.py` — speaker-disjoint split logic + tests.
- `src/data/simulated_users.py` — synthetic user stream generator.
- `data/library/pepper_v1.yaml`
- `data/library/dataset_to_pepper.yaml`

---

## 8. Baselines

All baselines share the same gesture library, the same `RobotInterface`, the same
data loaders. The only thing that varies is the selection logic. This is enforced
by giving every baseline a `Selector` implementation that conforms to the same
interface as our method.

### B1 — Vanilla SayCan port
Our own `SayCanAdaptSelector` with `adapt=None` and `weights.d=0`. This MUST equal
our method's `M0` configuration numerically — there is a unit test enforcing this
(`tests/test_selector_identity.py`). This is both a baseline and our central
ablation; it isolates exactly what Adapt + closed loop contribute.

### B2 — Wake et al. 2023 (GPT gesturing chat)
Reimplements the concept-mapping approach: LLM is asked to identify the
"conceptual meaning" of the utterance, then a lookup table maps concept → gesture.
No feasibility term. No adaptation. Open-loop.

### B3 — Torshizi et al. 2025 (LLM gesture selection)
Reimplements their LLM-as-selector method: LLM directly picks one gesture from the
library given the utterance and a structured prompt. No feasibility term. No
adaptation. Open-loop.

### B4 — Random / frequency prior
Sanity floor. Two flavors: uniform random, and frequency-weighted by gesture
occurrence in the training set. Any reasonable method must beat both.

**Files:**
- `src/baselines/saycan_port.py` (thin wrapper; the work is the unit test)
- `src/baselines/wake2023.py`
- `src/baselines/torshizi2025.py`
- `src/baselines/random_baselines.py`

---

## 9. Evaluation

### Metrics

#### Objective
- **Top-1 / Top-5 accuracy** vs. ground truth gesture.
- **Tag F1** — partial credit if selected gesture shares tags with ground truth.
- **Feasibility rate** — fraction of selected gestures that pass the Can check
  given the robot state. Baselines without a Can term should score worse here.
- **Diversity** — entropy of gesture distribution over a session. Pair this
  always with accuracy; a random policy maxes diversity but tanks accuracy.
- **Repetition rate** — fraction of consecutive same-gesture pairs in a session.
- **Latency** — wall-clock per selection (must be under the inter-gesture budget
  for real-time use on Pepper).

#### Long-term (the headline)
- **Adaptation curve** — accuracy plotted against session index, per user, then
  averaged. Fit a slope and report 95% CI. Positive slope with CI excluding zero
  is the claim.
- **Personalization gain** — accuracy of the per-user-adapted Adapt term minus
  accuracy with population-level Adapt only. Paired test across users.
- **Bandit regret** (Variant A specifically) — cumulative regret vs. an oracle
  that always picks the highest-eventual-reward gesture.

#### Subjective (optional, for paper polish — not required for v1 build)
- LLM-as-judge appropriateness on a held-out subset.
- ALWAYS report judge-vs-human agreement (Cohen's kappa) on a smaller human-labeled
  subset. Never present judge scores alone.

### Anti-gaming checks
- Diversity must be reported with accuracy.
- Feasibility must be reported with accuracy.
- A degenerate "always pick rest" policy must be detectable in the metrics; have a
  unit test that confirms it scores high on feasibility but tanks on accuracy and
  diversity.

### The protocol files
- `experiments/ablation_matrix.md` (exists) — defines what gets run.
- `experiments/metrics.md` (exists) — defines how it's measured.
- `experiments/long_term_protocol.md` — describes the simulated multi-session
  evaluation precisely: number of users, sessions per user, utterances per
  session, fixed seeds.

### Reports
Every run produces:
- `results/<run_id>/config.yaml` — exact config used.
- `results/<run_id>/metrics.json` — all metric values.
- `results/<run_id>/per_turn.parquet` — one row per turn with chosen gesture,
  per-scorer log-probs, reward, latency.
- `results/<run_id>/summary.md` — one-paragraph human-readable summary.

And appends a row to `results/leaderboard.json` (the global ledger).

**Files:**
- `src/eval/metrics.py` — every metric as a pure tested function.
- `src/eval/runner.py` — orchestrates a full eval run from a config.
- `src/eval/llm_judge.py` — the LLM-as-judge harness with mandatory kappa.
- `src/eval/leaderboard.py` — append-only leaderboard writer.

---

## 10. Implementation plan (file-by-file)

### Directory tree (target end state)

```
saycan-gesture/
├── CLAUDE.md                          # exists
├── README.md                          # exists
├── pyproject.toml                     # NEW: deps, ruff, mypy, pytest config
├── .claude/settings.json              # exists
├── configs/
│   ├── base.yaml                      # NEW: shared defaults
│   ├── B1_vanilla_saycan.yaml
│   ├── B2_wake2023.yaml
│   ├── B3_torshizi2025.yaml
│   ├── B4_random.yaml
│   ├── M0_identity.yaml               # must match B1 numerically
│   ├── M1_bandit.yaml
│   ├── M2_memory.yaml
│   ├── M3_adapter.yaml
│   └── ablations/                     # M1a, M1b, M1c, M1d
├── data/
│   ├── library/
│   │   ├── pepper_v1.yaml
│   │   └── dataset_to_pepper.yaml
│   └── cache/                         # gitignored
├── src/
│   ├── __init__.py
│   ├── say/
│   │   ├── __init__.py
│   │   ├── scorer.py
│   │   ├── llm_client.py
│   │   ├── cache.py
│   │   └── prompts/v1.txt
│   ├── can/
│   │   ├── __init__.py
│   │   ├── scorer.py
│   │   └── checks.py
│   ├── adapt/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── bandit.py
│   │   ├── memory.py
│   │   ├── adapter.py
│   │   ├── context_bucket.py
│   │   └── embeddings.py
│   ├── robot/
│   │   ├── __init__.py
│   │   ├── interface.py               # exists
│   │   ├── sim_playback.py
│   │   ├── library_loader.py
│   │   └── pepper/
│   │       ├── adapter.py
│   │       └── server.py              # Python 2.7
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py
│   │   ├── splits.py
│   │   └── simulated_users.py
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── saycan_port.py
│   │   ├── wake2023.py
│   │   ├── torshizi2025.py
│   │   └── random_baselines.py
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── runner.py
│   │   ├── llm_judge.py
│   │   └── leaderboard.py
│   └── pipeline/
│       ├── __init__.py
│       ├── selector.py                # exists
│       ├── feedback.py
│       ├── orchestrator.py            # NEW: top-level run loop
│       └── logging.py
├── experiments/
│   ├── ablation_matrix.md             # exists
│   ├── metrics.md                     # exists
│   └── long_term_protocol.md          # NEW
├── results/
│   └── leaderboard.json               # exists (skeleton)
└── tests/
    ├── __init__.py
    ├── test_selector_identity.py      # exists (the M0==B1 guardrail)
    ├── test_can_checks.py
    ├── test_adapt_cold_start.py
    ├── test_say_cache.py
    ├── test_data_splits.py
    ├── test_metrics.py
    └── test_baselines_parity.py       # confirms B1 == M0 end-to-end
```

### Build order (so nothing blocks anything)

1. **Foundation** — `pyproject.toml`, expand `RobotInterface`, `sim_playback.py`,
   `library_loader.py`, load `pepper_v1.yaml` with ~30 gestures.
2. **Data layer** — `loaders.py`, `splits.py`, `simulated_users.py`. Speaker-split
   test must pass.
3. **Can scorer** — pure rule-based, fully tested in isolation. Easiest scorer; do
   this first to get the pattern right.
4. **Say scorer** — `LLMClient` + prompt + cache. Test against a mock LLM client
   that returns deterministic scores; never burn API budget in unit tests.
5. **Adapt scorers** — implement Variant A (bandit) first, then B (memory), then C
   (adapter). Cold-start test must pass for all three.
6. **Feedback sources** — `ExplicitRatingFeedback`, `SimulatedEngagementFeedback`.
7. **Pipeline** — `orchestrator.py` tying everything together; `logging.py` for
   the per-turn parquet.
8. **Baselines** — B1 first (it's the identity check), then B2, B3, B4.
9. **Eval** — `metrics.py`, `runner.py`, `leaderboard.py`. The LLM judge can come
   last; it's optional for the v1 build.

### Coding standards
- Python 3.11+ for everything except the Pepper bridge.
- Type hints everywhere; `mypy --strict` clean on `src/`.
- `ruff` for linting and formatting.
- `pytest` for tests; every scorer and metric needs at least one unit test.
- No bare `except`. No `print` in `src/` — use the project logger.
- All randomness goes through a single `numpy.random.Generator` seeded from config.
  This is non-negotiable for reproducibility.
- Configs are YAML, loaded into typed dataclasses (use `pydantic` or `attrs`).

---

## 11. Configuration and reproducibility

### Every run is fully described by a single config YAML

Example (`configs/M1_bandit.yaml`):
```yaml
run_id: M1_bandit_seed42
seed: 42
data:
  dataset: beat2
  split: test
  library: data/library/pepper_v1.yaml
robot:
  adapter: sim_playback
selector:
  weights: {a: 1.0, b: 1.0, c: 1.0, d: 0.5}
  repetition_window: 5
say:
  backend: anthropic
  model: claude-3-5-sonnet-20241022
  prompt: src/say/prompts/v1.txt
  cache: data/cache/say_cache.sqlite
can:
  enabled: true
adapt:
  variant: bandit
  context_bucketing: keyword_v1
feedback:
  source: simulated_engagement
eval:
  protocol: long_term
  num_users: 20
  sessions_per_user: 10
  utterances_per_session: 50
```

### Run command
A single CLI entrypoint:
```bash
python -m src.eval.runner --config configs/M1_bandit.yaml
```
Produces `results/M1_bandit_seed42/...` and appends a row to the leaderboard.

### Reproducibility guarantees
- Config + seed + git commit fully determine the result.
- Config hash is stored in every output file.
- The LLM cache means re-running with the same config and same cache produces
  byte-identical results.

---

## 12. Scope and non-goals

### In scope for v1
- Discrete gesture selection from a fixed library.
- Three Adapt variants compared head-to-head.
- All baselines listed in §8.
- Simulated long-term protocol.
- Offline evaluation against labeled gesture data.
- `SimPlayback` execution; `PepperAdapter` built but hardware validation gated on
  hardware availability.

### Explicitly out of scope for v1
- **Gesture generation** (continuous motion synthesis). Stub the interface;
  do not implement.
- **Live human study.** The codebase must support running one (logging, ratings
  collection), but conducting it is a separate effort.
- **Multimodal feedback** (vision-based engagement detection). The
  `FeedbackSource` interface allows it but v1 only ships explicit + simulated.
- **Multi-robot embodiments.** Adapters for NAO / G1 are future work.
- **Real-time deployment optimizations.** Latency is measured but not the headline.

### Things to flag honestly in the paper
- The simulated long-term protocol is a proxy for real long-term deployment.
  This must be clearly stated.
- LLM-as-judge metrics need human validation.
- The Can term is rule-based; a learned naturalness scorer is future work.
- Pepper's mature SDK is Python-2-bound; the bridge architecture must be
  described honestly.

---

## 13. Definition of done for the v1 build

The codebase is "done" for v1 when **all of these are true**:

1. `pytest tests/ -v` passes, including `test_selector_identity.py` (M0 ≡ B1).
2. `mypy --strict src/` passes.
3. `ruff check src/ tests/` passes.
4. `python -m src.eval.runner --config configs/B1_vanilla_saycan.yaml` runs end to
   end on simulated data and produces a leaderboard row.
5. The same is true for `configs/M1_bandit.yaml`, `M2_memory.yaml`,
   `M3_adapter.yaml`, `B2_wake2023.yaml`, `B3_torshizi2025.yaml`,
   `B4_random.yaml`.
6. `test_baselines_parity.py` confirms B1 and M0 produce numerically identical
   per-turn logs given identical seeds.
7. The Pepper bridge runs against a NAOqi simulator (or stub) without crashing.

Once these are green, the system is ready for the experimental sweep — at which
point we add the Claude Code agent orchestration in a follow-up.

---

## 14. What to hand back when done

A pull request (or fresh clone) with:
- The full directory tree above populated.
- All tests passing.
- A short `BUILD_NOTES.md` documenting any assumptions made where this spec was
  underspecified, with a `# ASSUMPTION:` marker on each.
- The output of one full B1 run committed under `results/` as a sanity artifact.

That's the brief. Build to it.