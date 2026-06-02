# Build Notes

Assumptions and design decisions made where SPEC.md was silent. Every entry has a
`# ASSUMPTION:` marker in the corresponding source file.

---

## Step 3 — Can scorer

### ASSUMPTION: Joint range check uses mean log-prob across tracked joints

The SPEC says "each gesture declares joints_used" and "score degrades linearly with
how far out of range we are," but the `Gesture` dataclass has no per-gesture joint
range field. `check_joint_range` uses a universal neutral range of ±1.5 rad for all
joints; per-joint log-probs are averaged (not summed) so that gestures using more
joints are not systematically penalised. The floor is `log(0.01) ≈ -4.605`.

### ASSUMPTION: Three checks combined as sum of log-probs

`RuleBasedCanScorer` sums `check_pose + check_joint_range + check_time_budget`.
Summing log-probs is equivalent to multiplying the corresponding probabilities
(assuming independence), which matches SPEC §4.2 "three sub-scores are multiplied
(summed in log space)."

### ASSUMPTION: `log(0.01)` used as the infeasibility floor

Scores below `log(0.01) ≈ -4.605` are floored there; "effectively zero probability"
in linear space but not negative infinity, so the argmax in Selector remains
well-defined even for infeasible gestures.

---

## Step 1 — Foundation

### ASSUMPTION: RobotState is a mutable dataclass, not frozen

`RobotState` carries `joint_positions: dict[str, float]`. `dict` is unhashable, so a
`frozen=True` dataclass would raise `TypeError` on any hash attempt. The SPEC says
"frozen dataclass" only for `Gesture` (which uses `tuple[str, ...]` for its list
fields). `RobotState` and `ExecutionResult` are treated as value-type objects but only
`Gesture` and `ExecutionResult` use `frozen=True` (ExecutionResult has only hashable
fields). `RobotState` is a regular `@dataclass` (mutable, but treated as a snapshot in
practice).

### ASSUMPTION: Gesture uses tuple[str, ...] for joints_used and tags

The SPEC lists `joints_used: list[str]` and `tags: list[str]` for the `Gesture`
dataclass, but also says it is "frozen." Lists are unhashable, so `frozen=True` would
prevent using Gesture as a dict key or in a set. `tuple[str, ...]` is used instead.
The library loader converts lists from YAML to tuples at load time.

### ASSUMPTION: SimPlayback time-budget failure threshold is duration_s * 0.5

The SPEC states that the Can score "degrades" when `time_budget_s < duration_s` but
does not specify an exact failure cutoff for `SimPlayback.execute()`. The threshold of
`duration_s * 0.5` is used: below half the nominal duration, execution fails; at or
above it, the gesture runs at full nominal duration. A more nuanced model (proportional
duration scaling) is a v2 extension.

### ASSUMPTION: SayScorer Protocol returns dict[str, float] only

The SPEC describes a fallback mechanism (`say_fallback=True`) but is silent on whether
the scorer itself should expose it or whether it should be handled by the caller. The
`SayScorer` protocol returns `dict[str, float]` only; the `Selector` wraps the scorer
call in `try/except` and sets `say_fallback=True` in `SelectionRecord` if the scorer
raises. This keeps the Protocol clean and puts resilience in one place.

### ASSUMPTION: AdaptScorer.state_dict() uses dict[str, Any]

The three Adapt variants store very different internal structures (Beta posteriors,
embedding indices, neural network weights). `dict[str, Any]` is used as the
checkpoint type for maximum flexibility. Callers must verify the content type when
loading.

### ASSUMPTION: Standard logging used in selector.py until pipeline/logging.py exists

`src/pipeline/logging.py` (the project logger) is a Step 7 artifact. For Step 1,
`logging.getLogger(__name__)` is used in `selector.py` for the say-fallback warning.
This will be replaced with the project logger when Step 7 lands.

### ASSUMPTION: pepper_v1.yaml uses Pepper's documented NAOqi 2.x joint names

Joint names follow NAOqi 2.x conventions (e.g., `LShoulderPitch`, `RElbowRoll`,
`HeadYaw`). The `naoqi_animation` paths follow the standard Pepper animation library
format (`animations/Stand/Gestures/<name>`). Paths that don't match a real Pepper
animation exactly will need to be mapped by the Pepper adapter; the YAML is the
semantic source of truth, not the animation path.
