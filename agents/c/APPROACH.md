# Causal World Model via Interventions (Builder C)

## The thesis in one paragraph

In an interactive environment, **actions are interventions**. That single
observation dissolves the usual boundary between "learning from data" and
"acting": every step the agent takes is a `do()` operation on the world's
causal mechanisms, and the resulting state change is interventional data —
not a correlational sample. An agent that treats its experience this way can
learn a structural causal model (SCM) of the game: which variables each
action's mechanism controls, how variables influence each other over time,
and which mechanisms stay invariant when the context (level, layout) shifts.
Planning is then *simulated intervention*: forward-sample the SCM under
candidate `do()` sequences and pick the sequence predicted to reproduce the
variable configuration that previously caused a win. Curiosity falls out
naturally — the most valuable intervention is the one whose mechanism is
most uncertain.

## Why causality is the right formalism here

1. **Actions ARE interventions.** Reinforcement learning treats
   `(s, a, r, s')` as samples from a joint distribution and needs enormous
   data to separate "what my action caused" from "what merely co-occurred".
   The do-calculus starts from the right primitive: `P(Y | do(A=a))` is
   *defined* by acting. There is no confounding of an agent's own actions —
   the agent chose them.
2. **Invariance separates mechanism from correlation.** A correlation like
   "score increases when the green blob is large" may hold on level 1 and
   break on level 2. A mechanism — "pressing LEFT adds one green sprite" —
   holds wherever the game's code is the same. We keep an edge only if its
   effect survives context shifts (level index, which objects exist). This
   is what lets knowledge transfer across levels instead of being
   re-learned.
3. **Credit assignment needs states, not just actions.** When death follows
   many steps after the fatal move, blaming the last action is wrong. We
   learn *doomed states* — variable configurations after which attempts
   reliably die — and penalize interventions predicted to lead into them.
   The killer move is identified by its predicted consequence, not by
   temporal proximity to the death.

## Architecture

```
frame (64x64 int8)
  -> variables.py   : 17 binned object-level variables
  -> causal.py      : interventional data -> causal graph (invariance-tested)
  -> planner.py     : win-direction + greedy SCM simulation + beam search
  -> agent.py       : episode loop, intervention enumeration, budgets
```

### 1. Variables (`variables.py`)

Each frame is parsed into 17 binned variables: number of distinct colors,
total foreground pixels, min distance between the two most prominent
colors, global largest-object centroid, and per color-slot (6 slots,
first-seen stable identity) pixel count and object count (4-connected
components). Six slots are kept so click interventions cover 5+-color
games; per-slot position was traded for the global centroid to stay under
the ~20-variable budget. Binning lesson learned the hard way: coarse bins *destroy the intervention signal* — a 270-pixel change
left every variable unmoved. Bins are now fine enough that typical
single-action effects register (object counts are barely binned at all:
one action often adds exactly one object). Sparsity from fine bins is
handled by pooling in the discovery statistics.

### 2. Interventions (`agent.py`)

The intervention set is enumerated fresh each step from `available_actions`:
each simple action id is one intervention; `ACTION6` (click) is
parameterized by **clicked-object identity** — `('C', k)` clicks the topmost
pixel of color-slot k's largest object, `('CB',)` clicks background as a
null control. Object-anchored (not coordinate-anchored) clicks are what make
"click the green thing" a stable intervention across frames.

### 3. Causal discovery (`causal.py`)

For each `(intervention, variable)` pair we maintain incremental statistics:
`P(V changes | do(I))` vs the same under all other interventions (the
contrast *is* the causal effect — no separate observational baseline
needed, since all data is interventional). An edge is kept if the effect
is strong **and invariant**: per-context effect estimates (context =
level index × color-presence mask) must agree in sign and roughly in
magnitude; single-context edges are kept provisionally at reduced
confidence. Variable→variable edges come from temporal co-change:
`P(B changes | A changed) - P(B changes | A unchanged)` within a
transition. All statistics are incremental; the whole thing is numpy-only.

**Model selection by invariance.** Three candidate variable subsets
(full / mid / min) are scored by time-split one-step prediction accuracy,
penalized by the train/test gap — the mechanism that predicts best across
time (a proxy for across-context invariance) wins, ties broken toward
parsimony.

**Doom states.** Because the fatal move often precedes death by many steps,
we additionally learn `P(attempt dies | variable = value)` over *states*,
marking the ~25 states before each death. Interventions whose SCM-predicted
post-state is doomed are penalized. This is the credit-assignment fix that
stopped the agent re-trying lethal moves whose blame had landed on an
innocent final action.

### 4. Planning (`planner.py`)

**Goal discovery.** When `levels_completed` increments, we record the
*win direction*: the signed change of each variable from the level-start
configuration to the pre-win configuration (an exponential moving average
over wins). Using level-start → pre-win, rather than the raw winning
transition, matters: the post-win frame already shows the reloaded next
level, which would invert the learned direction. The goal is a *direction
in variable space*, transferable across levels — not a memorized state.

**Control.** Greedy one-step simulation on the SCM:
`score(I) = Σ_V w_V · sign(win_dir_V) · E[ΔV | do(I)] · conf(I→V)`
`− 6·P(died|I) + 4·P(win|I) − 5·doom(predicted_post(I)) + UCB(I)`.
Before the first win there is no goal, so the agent does **active causal
learning**: UCB over interventions plus bonuses for expected variable
change and raw pixel-change magnitude (the latter catches rearrangements
that conserve counts). If the raw state stalls for 14 steps with a goal in
hand, a beam search (depth 5, width 48) forward-samples the SCM for a
sequence maximizing win-direction progress.

### 5. Environment gotchas handled (verified against engine internals)

- `obs.frame` is a *list*; the winning action's returned frame may already
  show the next (reloaded) level.
- `ACTION6` coordinates must go through `step(data={"x":..,"y":..})`.
- **RESET is not free**: it increments the scorecard action counter, and —
  worse — a RESET when the game's internal `_action_count == 0` triggers a
  **full reset that wipes the score**. This happens right after a level win
  (the next action loads the new level and zeroes the counter). The agent
  tracks this exactly and RESETs only when mandatory (after `GAME_OVER`).
- Failed attempts' actions count toward the level total, so exploration is
  budgeted per level (`min(4·baseline+10, 600)` actions) and per game.

## What the agent is good at, and what it is not

It is good at games whose causal structure is *learnable from
intervention*: discover which actions move which objects, avoid doomed
states, and hill-climb the win direction. On the two toy games (move-left /
click-left puzzles) it goes from zero knowledge to perfect play (5/5
levels) in under 200 actions, with an explicit, inspectable causal graph.

It is not a puzzle solver. Real ARC-AGI-3 games are obfuscated, multi-step
puzzles where the win requires a specific long sequence and the only
feedback is the final `levels_completed` increment. Curiosity-driven
intervention testing cannot find a 6-click combination in a 34-action
budget, and the agent honestly reports 0 levels there. The framework's
claim is narrower: *when* the mechanism is identifiable from interaction,
causality (interventions + invariance + credit assignment via doomed
states) is the right language for learning and exploiting it — and the
learned model transfers across contexts instead of being re-fit per level.
