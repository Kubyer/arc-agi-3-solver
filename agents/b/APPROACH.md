# Builder B: Object-Centric World Model + Curiosity Exploration

*ARC Prize 2026, ARC-AGI-3 paper track — agent design document.*

## 1. Thesis

Interactive ARC-AGI-3 games are won or lost on **exploration**, not on policy
sophistication. The action semantics are unknown, the reward is sparse (level
complete / game over / nothing), and RESET is free. We therefore build an
agent around two ideas:

1. **An object-centric inductive bias (core knowledge).** Humans parse these
   scenes as *objects that move, appear, and disappear* — not as pixel arrays.
   Our agent segments every frame into color-connected components, and both
   its world model and its exploration archive operate on object-level
   descriptions, never on raw pixels.
2. **Curiosity as the objective.** In the absence of reward, the agent seeks
   *novel predictable structure*: actions whose effects are unknown, and
   states the world model predicts will be new. Exploration is organized
   Go-Explore style (archive of interesting states + return via free RESET),
   which turns blind thrashing into systematic coverage.

The agent is offline, CPU-only, NumPy/SciPy only. No learned weights, no
neural networks, no LLMs — the "world model" is a table of discrete
object-transition statistics.

## 2. Perception: objects, not pixels

`objects.py` segments each 64×64 frame:

- **Background** = the most frequent color (chosen empirically per frame;
  it is *not* always color 0 — several games use colored backdrops).
- **Objects** = 4-connected components of each non-background color
  (`scipy.ndimage.label`).
- Each object is summarized as `(color, size, coarse position, bounding box)`.
- A **state signature** = the sorted tuple of
  `(color, coarse_y, coarse_x, size_bin)` over objects, with 8-pixel spatial
  bins and logarithmic size bins. This is the discrete "cell" used by the
  archive and the model — coarse enough to generalize across levels, fine
  enough to distinguish meaningfully different configurations.

Nothing downstream ever sees raw pixels.

## 3. The world model: discrete object transitions

`model.py` maintains, per discrete action-key, a small statistics table:

- appearance / disappearance counts per object class,
- same-color centroid displacement vectors (for movement),
- mean appearance positions,
- no-op frequency, fatality (game-over) frequency.

From these it can **predict** the coarse outcome of an action: which objects
will move where, which new objects will appear. Prediction is deliberately
coarse (it predicts signatures, not pixels) — its job is to support
curiosity ("will this lead somewhere new?") and short-horizon planning,
not photorealistic simulation.

A key design point: **action-keys abstract away coordinates.** For click
games (`ACTION6`), the key is not "click (x, y)" but **"click the largest
object of color c"**, resolved to the object's live centroid at execution
time. The key space is therefore small (one key per color present, not
4096 pixels), and — crucially — learned knowledge transfers across levels
even when sprites change scale or position, because it is indexed by
*color role*, not screen location.

## 4. Curiosity: the exploration objective

`agent.py` scores every legal action-key each step. The score combines:

- **Global novelty** (+1.2): has this key *ever* been tried? (Grid-click
  probes are deliberately deprioritized at +0.8.)
- **State novelty** (+2.0): has this key been tried *from this state*?
- **Predicted novelty** (+1.5): does the world model predict a state (or an
  object appearance) the archive has never seen?
- **No-op penalty** (−1.0): the model says this usually does nothing.
- **Fatal penalty** (−2.0): this key caused game-overs before.
- **Valence** (±0.8–1.2): colors whose objects *net-appeared during a
  successful level* are attractive; colors that net-appeared before a loss
  are repulsive. Valence is learned, not hardcoded.
- **Solution transfer** (+3.0): action sequences that solved earlier levels
  are replayed verbatim on new levels (many games repeat mechanics).
- **Tabu**: trajectories that ended in game-over or in a detected
  world-model *surprise* are never repeated.

**Surprise detection** deserves emphasis: before each step, the agent asks
the model what it expects. If a well-understood action produces an outcome
that contradicts the prediction *in a valence-relevant way* (expected good
objects didn't appear, or bad ones did), the agent concludes the level is in
a hidden doomed state, aborts the attempt immediately (RESET is free), and
taboos the trajectory. This converts silent doom into a learning signal.

## 5. Go-Explore organization: archive + free RESET + model BFS

`archive.py` keeps every visited signature cell with its first trajectory,
visit counts, and tried/untried actions per cell. When the policy's best
score drops below threshold, the agent:

1. picks the **least-visited frontier cell** (a cell with untried actions),
2. RESETs (free) and **replays the stored trajectory** to return to it,
3. continues exploring from there.

A bounded **BFS over the learned transition model** (depth ≤ 3, ≤ 200 nodes)
plans action sequences toward predicted-novel frontier cells, so
exploration is model-directed, not just ε-greedy.

## 6. What worked

- **Toy games are solved perfectly.** On `bt11` the agent completes 5/5
  levels in exactly the 72 baseline actions (RHAE 100 per level); on `bt33`
  it completes 5/5 in 78 actions. The click abstraction, solution transfer,
  and valence learning all fire as designed: level 0 is solved by curiosity,
  later levels by replaying and adapting the discovered strategy.
- **The world model learns real structure.** On real games it correctly
  learns e.g. "clicking color-4 objects grows them", no-op rates per action,
  and which actions are fatal — from a handful of samples.
- **No crashes or API violations** across all 25 games and both toys:
  `obs.state` is always checked before `obs.frame`, `available_actions` is
  re-read every step, ACTION6 coordinates go through `data={"x","y"}`,
  RESET is never counted as a scored action.

## 7. What failed, and why

- **Zero real ARC-AGI-3 levels solved.** The agent explores systematically
  (hundreds of distinct states per game) but cannot solve a single real
  level. The reason is fundamental, not implementational: **curiosity
  discovers *how* the world works, but not *what the goal is*.** Real
  ARC-AGI-3 levels require inferring the goal from the scene (e.g. which of
  several growing objects is the target, what configuration counts as
  "done"). Our agent has no goal-inference mechanism — no reward gradient,
  no demonstration, no visual question-answering — so it wanders the state
  space without ever recognizing the target. The toy games are solvable
  precisely because their goal ("click the odd/good one repeatedly") is
  discoverable by valence alone.
- **Coarse signatures blind the agent to fine progress.** The 8-pixel /
  log-size coarsening means pixel-by-pixel growth (as in `vc33`) often
  produces *no* signature change, so the curiosity signal flatlines even
  while the agent is doing something meaningful. Finer signatures would
  explode the archive; this is an unresolved tension.
- **Frontier returns have limited value without a goal.** Go-Explore's
  return mechanism assumes you know which cells are *promising*. With only
  novelty as the criterion, returns mostly re-tread known space.

## 8. Limitations and next steps

The experiment isolates a clean finding: **object-centric curiosity is
sufficient for learning environment dynamics and for solving
valence-discoverable tasks, but goal inference is the missing half of
interactive ARC reasoning.** The natural next step is a goal module — e.g.
inferring target configurations from level-intro frames, or learning a
success classifier from the sparse level-complete signal — plugged into the
same object-centric representation. The world model and archive are already
the right substrate; they just need something to aim at.

## 9. Reproducibility

```
code/agents/b/
├── __init__.py      package exports (run_episode, ObjCuriosityAgent)
├── objects.py       frame -> object list + state signature
├── model.py         per-action object-transition statistics + prediction
├── archive.py       Go-Explore cell archive
├── agent.py         curiosity policy, surprise detection, returns, BFS
├── goals.py         goal-hypothesis module (visual targets, predicates,
│                    hypothesis testing over the learned model)
├── agent.py         curiosity policy, surprise detection, returns, BFS,
│                    goal-hypothesis mode switch
├── benchmark.py     per-game runner + RHAE scoring -> RESULTS.md
├── APPROACH.md      this file
└── RESULTS.md       benchmark numbers (generated)
```

Run: `.venv/bin/python agents/b/benchmark.py [game_id ...]`
(default: all 25 real games + bt11/bt33; ~1–3 min per game, seed 0).

## 10. Goal inference module (iteration 2)

Section 8 diagnosed the missing half: curiosity learns dynamics but cannot
infer goals. `goals.py` adds a surgical goal-hypothesis layer on top of the
unchanged curiosity machinery; `agent.py` gains a mode switch and a few
lifecycle hooks, nothing else.

**Visual target detection.** No extra probe actions are spent: the module
harvests the parsed frames curiosity already collects. An object is *static*
if an object of the same color stays at the same centroid with the same size
in ≥85% of observed frames (per-object, not per-color — a color can have
both static and moving instances; frames 0–1 are excluded because a
mid-game level change can leave one stale frame in the pipe). Hollow
rectangular outlines — a color forming a ring (large, low bbox fill ratio,
empty interior) — are detected as target zones; their interiors are prime
goal regions. Large static objects become regions too.

**Hypothesis generation.** Eight predicate templates over object relations,
parameters bound from the current scene (colors, region rects — never
absolute pixel scripts), ranked by prior, static-region-involving first:
`inside_outline` (1.00) > `fill_outline` (0.90) > `overlap_static` (0.80) >
`centroid_region` (0.75) > `touch_static` (0.60) > `cover_region` (0.55) >
`eliminate` / `clear_region` (0.50). Hypotheses already true in the live
frame are refuted for free (a true predicate that didn't complete the level
cannot be the goal).

**Hypothesis testing.** The top untried hypothesis is converted to an action
sequence by BFS (depth ≤ 5, ≤ 600 nodes) over `model.predict_objects` — the
*learned* transition model — and executed with live click re-resolution.
Verdict rule: *refuted* only if the predicate was actually achieved without
completing the level; if the plan failed to achieve the predicate the test
is *inconclusive* (bad plan, not necessarily a bad hypothesis) and the
hypothesis stays untried. Game-over or a world-model surprise during a test
refutes it. Refuted hypotheses are never retried on the same level.

**Transfer.** Confirmed *templates* (not absolute bindings) transfer across
levels via a prior boost (+0.5 per confirmation), hooking into the existing
solution-transfer machinery (which continues to replay winning trajectories
verbatim). The scoreboard (confirmed/refuted/untried) is per-level; the
template win counts persist across the game.

**Mode switch.** `agent.py` enters goal-hypothesis mode when curiosity stalls
(no new archive cells for K=150 steps) or every M=600 actions (120-step
cooldown between evaluations), provided the transition model is mature
enough to plan with (≥2 keys observed ≥3 times) and at least 6 frames have
been seen. Tests cost ≤5 actions each. On the toy games the switch never
fires (they solve in <150 actions), so toy behavior is bit-identical.

**Cost.** One `parse_frame` per goal step plus one per finished test, and a
bounded BFS every ~120 steps — negligible next to the curiosity loop.

## 11. Workstream 1: strengthening the planner (goal discovery + long sequences)

This section records the workstream-1 changes to the goal module, aimed at
the diagnosed bottleneck: discovering goals and composing long unrewarded
action sequences. All four priority directions were implemented; the toy
gate (`bt11`: 5/5, 72 actions, W-RHAE 100.0; `bt33`: 5/5, 96 actions,
W-RHAE 78.3) stays bit-identical after every change.

**(a) Distributional transitions.** `model.sample_objects(key, objs, rng)`
draws from the empirical per-key outcome distribution instead of the mean:
per-color displacement outcomes (including zeros), disappearance
probabilities, and recorded appearance positions. A synthetic check shows a
key with outcomes `[10,12,14,16,18,20]` now samples all six instead of only
the mean 15. `predict_objects()` was also switched to a cached mean lookup
(`_mean_moves_cached`, keyed by observation count) — a pure performance fix.

**(b) Deeper heuristic search.** `plan_beam()` replaces the deterministic
mean-displacement BFS (depth 5, 600 nodes) with best-first beam search over
*sampled* model outcomes: depth 12, beam 24, 3 samples per expansion, 8000
node budget. States are scored by smooth predicate progress
(`GoalModule.progress`, per-template [0,1]) plus a novelty bonus, with a
fine-grained visited set (see (c)). Candidate plans that achieve the
predicate are validated by Monte-Carlo rollouts (`_validate_plan`,
3/5 successes required) to reject lucky sampled branches. On a synthetic
navigation task the beam finds a 9-step plan that the depth-5 mean BFS
cannot. Up to 3 hypotheses are tried per evaluation (`N_HYPS_PER_EVAL`);
unplannable ones are skipped via plan-fail counters rather than burning the
evaluation.

**(c) Pixel/grid-level precision.** The planner's visited set uses
`fine_hash()` (2px position bins) instead of the archive's coarse 8px
signature, so 1px movements are distinguished during planning. (The
original "pixel/grid-level planning" direction is subsumed: the precise
hash plus distributional sampling gives the planner the resolution it
needs without a separate grid planner.)

**(d) Goal-directed behavior from frame 1 + sustained pursuit.** Every
`GOAL_EVERY=12` decisions, the agent runs a short beam search toward the
top *region-grounded, manipulable* hypothesis and executes the result:
a validated full plan becomes a formal test; otherwise a partial plan
with predicted progress gain ≥ 0.15 is executed as a *sustained pursuit*
(up to 6 steps), composing the multi-step sequences that single
best-effort steps cannot. Region-grounding (a detected outline/static
target) keeps this a no-op where no visual target exists, preserving
toy bit-identicality. The evaluation path (stall/periodic) uses the same
mechanism with the full node budget.

**Sharper hypothesis filtering.** Two fixes proved critical. (1) A dedicated
`goal_rng` (`random.Random("goals-<seed>")`) separates goal decisions from
the curiosity RNG — without it, planner sampling perturbed toy behavior
even when no goal action was taken. (2) Region colors are excluded from
the mover set: a color that is mostly a large static structure (wall,
border) is never a "mover", even if a small fragment shifts. Before this
fix the agent wasted pursuits trying to move walls (e.g.
`overlap_static(mover=5, target=10)` on ar25, both static); after, it
pursues the actual player (`mover=4`).

**Hindsight credit.** `GoalModule.hindsight()` confirms pursued hypotheses
whose predicate holds in the level-winning frame (25-action window),
transferring confirmed *templates* across levels via prior boost. Pursuits
(single-step or sustained) never refute: only a formal test whose
predicate was achieved without completion refutes.

**Result.** On the dev subset (`ar25 g50t lp85 lf52 vc33`, seeds 0,1,2,
4000 actions / 180s): baseline `GOALS_OFF=1` solves 0/15 levels (mean
W-RHAE 0.0); the workstream-1 agent also scores 0/15 (mean W-RHAE 0.0),
with substantial planner overhead (e.g. lf52 wall 97.0 ± 9.5s goals-on vs
29.7 ± 2.0s goals-off in paired probes). A measurement flaw was found and
fixed: `finish()` reported the *final* `levels_completed`, which returns to
0 after later resets; the agent now tracks `levels_ever` (max ever seen)
plus the goal state at each winning decision. Paired goals-on / `GOALS_OFF`
probes on lf52 (seeds 0,1,2) show both variants complete L0 (goals-on at
actions 22/32/37, off at 22/32/31), with the winning decision never
goal-directed (`from_goal=false`, no test/pursuit in flight; in seeds 0-1
the completion precedes the first goal action). The pre-workstream-1 full
run (`bench_all.log`) also shows lf52 L0 completing at action 22. The
earlier claim that sustained `inside_outline` pursuit "solves lf52" is
retracted: the completions are baseline-curiosity completions the planner
does not cause, and none persist into a full-game score. Net: **zero
measured dev-subset improvement over baseline**, so no 27-game run was made.
See `RESULTS_W1_PLANNER.md` for full tables.

**Biggest weakness.** The eight hand-written templates do not cover all
game types: games without detectable visual targets (g50t: zero outlines)
get no interleave benefit, and maze navigation (ar25) needs
obstacle-aware planning the global per-key displacement model cannot
provide. The planner optimizes hand-written predicate progress, not a
learned success signal.

## 12. Workstream 2: MDL program world-model graft (agent A's KeyModel)

**Status: partial graft — carried forward with a narrow scope.**

### What was grafted

Agent A's MDL program-induction world model (`agents/a/world_model.py`) is
loaded by file path into `agents/b/mdl_model.py`; `agents/a/` was not
modified. B keeps one A `KeyModel` per B action key (keyboard `aN`, object
click `o<color>` with actual click coordinates, grid click `gN`, plus a
shared click fallback), learns from coarsened 32×32 before/after frames,
and tracks falsifications, stable ("mature") programs (8+ uses), and optional
diagnostics. Env switches: `MDL_NO_PREDICT=1` (pure-B statistics behavior),
`MDL_NO_BONUS=1` (no falsification bonus), `MDL_NOOP_PENALTY=1` (re-enable
the rejected no-op penalty).

### Two failed designs, one surviving design

1. **Naive MDL→fine-object prediction (rejected).** Predicting 64px object
   signatures from 2×-subsampled program grids broke the bt33 toy gate.
   2× subsampling + nearest-neighbor upscaling destroys thin/disconnected
   structure (an ar25 prediction with 12 grid errors vs identity's 54
   reparsed into 6 objects vs 12 true ones). Fine-signature accuracy of MDL
   predictions is ≈ 0.00 — B's original statistics stay the sole predictor
   for objects, BFS/planning, and valence.
2. **Grid-exact no-op penalty (rejected).** Penalizing actions whose coarse
   prediction changed ≤2 cells regressed bt33 to 4/5 levels / 4000 actions:
   a coarse "no-op" can hide a sub-32×32 fine-space effect the subsampling
   blinds the program to. Disabled by default.
3. **Surviving design (kept).** B statistics keep all prediction duties.
   MDL contributes only two conservative channels: (a) predicted-novelty
   scored in coarse 32×32 signature space, only for mature keys (never
   upscaled to fine space); (b) a bounded falsification curiosity bonus
   (+0.3 for mature keys whose program was falsified within the last 10
   uses, plus a shared-click fallback bonus).

### What the measurements say

- **Prediction genuinely improves, but heterogeneously.** Multi-seed
  grid-Hamming error per transition (final config, seeds 0–2): mature MDL
  programs beat the identity baseline by **1.38× on average** — 8.83× on
  vc33, 1.57× on g50t, 1.1–1.2× on ar25/lp85, and 0.98× (null) on lf52,
  whose non-stationary per-level dynamics defeat the programs.
  Coarse-signature accuracy is only at identity parity (0.67 vs 0.69) — the
  programs get *closer* (graded error) without matching exactly (binary
  signature) — so the coarse-novelty signal is noisy; it regularizes
  exploration rather than predicting precisely.
- **Toy gate holds bit-identical:** bt11 5/5, 72 actions, W-RHAE 100.0;
  bt33 5/5, 96 actions, W-RHAE 78.3.
- **Dev-subset game delta is small but causally attributed.** Same-session
  seeds 0–2, goals on: baseline 6/15 transient L0 completions vs graft 7/15;
  the only delta is ar25 seed 1 L0 at action 201 (baseline: 0 in 1201).
  Extended ar25 seeds 3–4: baseline 0/2, graft 1/2 (seed 3, action 184) →
  2/5 vs 0/5. Both completions are curiosity-driven (`from_goal=false`) and
  post-maturity. Attribution: the winning decision would also win under
  stats-only scoring (graft−stats = −0.034) — **the graft shaped the
  exploration path, not the final choice**; baseline/graft trajectories are
  identical for 31 decisions, diverging first on coarse novelty (decision
  32: untried `o5` over the mature-program-predicted-visited `a4`) and then
  on the falsification bonus (decision 41). `MDL_NO_BONUS=1` kills the
  seed-1 completion, so the bonus is load-bearing on that path.
- **No persistent score improvement.** W-RHAE is unchanged at the dev-subset
  level (all completions are transient L0s); no 27-game run was warranted.
- **Compute is fine:** ≈ 0.7 ms per coarsen, ≈ 0.035 s amortized refit;
  graft runs ≈ 10–20% slower per game, inside the 180 s budget.

### Carry-forward

Keep the MDL world model as a passive forward model + falsification signal
with the two conservative consumption channels. Do not attempt fine-object
prediction from subsampled grids without a topology-preserving
representation; the grid-exact no-op penalty direction is dead. Full
tables: `RESULTS_W2_MDL.md`.
