# Self-falsifying program world-models (MDL search) — Approach

Agent A for the ARC Prize 2026 paper track (ARC-AGI-3). Fully offline,
CPU-only, NumPy-only. Entry point: `MDLAgent.run_episode(env)` in
`mdl_agent.py`.

## 1. Idea

Learn, per action, a small *program* that predicts the next observed frame.
Choose programs by the **Minimum Description Length** principle: the best
program is the one that minimizes

```
S(rule) = E(rule) + λ · L(rule)
```

where `E` is the total cellwise prediction error of the program on a buffer
of recent transitions and `L` is the program's description length.
Programs that stop predicting well are *falsified* (Popper-style) and
re-searched. A receding-horizon planner rolls out the surviving programs and
explores to maximize novelty and prediction-error curiosity, which both
feeds the falsification loop with informative data and stumbles into level
completions.

## 2. State and action representation

- A frame is the last `(64, 64)` int8 observation, coarsened by 2×
  subsampling to a **32×32 grid** `s ∈ {0,…,15}^{32×32}` (`GRID = 32`).
- The background color `bg` is the grid mode.
- One world-model is learned per **action key**:
  - keyboard actions: `key = action_id` (1–5);
  - clicks (`ACTION6`): `key = (6, qx//4, qy//4)` — an **8×8 region** of the
    32×32 grid, so click effects generalize spatially;
  - plus a **global click model** `key = (6, "any")` shared across regions,
    used as a fallback predictor for regions with no data yet.
- The agent only ever uses `obs.available_actions`; `ACTION6` coordinates
  are passed as `env.step(action, data={"x":…, "y":…})` on the 64×64 grid.

## 3. The program hypothesis space (`Rule`)

A rule predicts `s' = predict(s, click)`:

```
predict(s):  base = apply_mode(s)            # motion part
             for (x,y)->c in delta: base[y,x] = c   # sparse overwrite
```

**Modes** (`apply_mode`):

| mode   | meaning |
|--------|---------|
| `none` | identity: `base = s` |
| `all`  | whole-frame translation by `(dx,dy)`; cells shifted in from outside are filled with `bg` |
| `c`    | translate only cells whose color is in `colors` by `(dx,dy)`; vacated cells are filled with a learned `fill` color |
| `goto` | move the foreground (non-`bg`) centroid to the click position |

The **delta table** is a sparse set of absolute overwrites
`(x, y) → color`, catching effects the motion part cannot express
(blinking cells, appearing/disappearing sprites, counters).

### 3.1 Description length

```python
def length(self):
    L = len(self.delta)
    if self.mode in ("all", "c") and (self.dx or self.dy):
        L += 2                       # the shift vector
    if self.mode != "none":
        L += 1                       # the mode selector
    if self.mode == "c":
        L += len(self.colors)        # which colors move
        if self.fill != self.bg:
            L += 1                   # non-background fill
    return L
```

### 3.2 Data term

For a buffer of `n` pairs `(s, s2, click)`:

```
E(rule) = Σ_pairs  Hamming(predict(s, click), s2)     # cell errors, 0..1024
S(rule) = E(rule) + λ · L(rule),   λ = 5.0 (LAM_DEFAULT)
```

So one description token costs as much as 5 mispredicted cells.

## 4. Searching for the MDL-best rule

Search happens per action key inside `KeyModel`, over a buffer of up to
`BUF_CAP = 24` recent *frame-changing* transitions (no-change transitions
update only the error statistics, keeping the buffer informative).

**4.1 Candidate generation (`fit_rule`).** Enumerate `(mode, dx, dy, colors)`:

- `all`: brute-force all shifts in `[-3, 3]²`, keep the 2 with lowest error
  on the last 6 pairs (camera motion).
- `c`: color sets = the top-3 singleton colors from `move_colors` (colors
  covering >15% of changed-cell mass, excluding `bg`) plus the full set;
  shifts from centroid estimation (`est_shift`: centroid of vanished cells
  of those colors → centroid of appeared cells) ± a 5×5 local search; keep
  the 2 best per color set.
- Always include `none` (identity) and, for clicks, `goto`.

**4.2 Exact sparse delta (`_fit_delta`).** For a fixed
`(mode, dx, dy, colors, fill)`, delta entries sit on distinct cells, so their
error contributions are independent. For each cell `(x,y)` and color `c`:

```
reduction(x,y,c) = fix − hurt,   fix  = #{pairs: base[y,x] ≠ s2[y,x] = c}
                                     (cell repaired)
                                 hurt = #{pairs: base[y,x] = s2[y,x] ≠ c}
                                     (cell broken)
```

Keep the argmax color per cell iff `reduction > λ` (the entry pays for its
own description) and it fires on at least half the pairs
(`fix ≥ max(2, (n+1)//2)`). Computed fully vectorized with `np.add.at` —
a 13-transition fit dropped from ~4.4 s to ~0.075 s with this change.

**4.3 Evolutionary refinement (`refit`).** A population of `POP_SIZE = 3`
programs is maintained per key. On refit: take the `fit_rule` result, add
`N_MUTANTS = 8` mutants (shift nudge, delta entry add/drop, drop whole
delta, mode switch, color-set re-estimation, background re-estimation,
small random color-shift) and crossover children of the top-2 (delta
intersection and union), then keep the 3 unique programs with lowest `S`.

## 5. The falsification loop

Each `KeyModel` tracks an EMA (α = 0.1) of its single-step prediction
error, updated on *every* transition (including no-change ones, via
`note_prediction`). A refit is triggered when the current program is
**falsified** —

```
err > max(12.0, 2.0 × ema_err)
```

— or on a schedule: the first time 3 pairs are buffered, then every 15th
informative transition. The population (not just one program) is kept so a
refit can resurrect a previously good hypothesis when the environment
changes (e.g. a new level with different dynamics).

Game overs are tracked per key; a key with ≥2 game overs and a >40% rate
is marked **dangerous** (also inherited from the fallback click model).

## 6. Planning under the learned programs

Receding-horizon beam search (`planner.py`): `DEPTH = 4`, `WIDTH = 16`,
branching over legal actions (clicks branch over `N_CLICK_CANDS = 10`
proposed positions). Each 1-step reward is:

```
r = novelty(pred) − 0.02
    + win_bonus(key, g)
    + 1.5 · √(log T / (1 + n_key))          # UCB over action keys
    + 2.0 · √(min(ema_err, 256) / 32)       # prediction-error curiosity
    − 5.0 · 1[key dangerous]
    + N(0, 0.05)
```

- **novelty** = `1/√(1 + visits(hash16(pred)))`: hash of the 16×16
  subsample; visits counted per level.
- **curiosity**: actions whose effects the model still predicts poorly
  attract exploration until learned; perfectly predicted no-ops
  (dead clicks) get ~0.
- **win_bonus**: +2.0 for reusing an action key that previously incremented
  `levels_completed`, +6.0 if the pre-frame hash also matches.
- `T` = total key tries; `n_key` = tries of this key.

**Action selection mixture** (per decision): 10% start an *exploration
burst* (repeat one action 4–10 times — how sustained effects like corridors
or repeated button presses are discovered; click bursts aim at candidate
positions, not random pixels); 3% uniform random; 35% 1-step
novelty-greedy; otherwise the beam plan. When a level was previously won,
30% of decisions replay the winning action suffix (last ≤80 actions):
levels of a game usually share mechanics.

**Click candidates**, in order: centroids of small (≤120-cell) non-`bg`
connected components (likely buttons), smallest first; previously
*effective* click cells; untried non-`bg` cells with 8×8 regions that
already absorbed ≥2 dead clicks deferred to the end.

**Episode bookkeeping**: wins are detected from `levels_completed`
increments (the new level may load on the next action, so per-level action
counts are attributed carefully); `GAME_OVER` → free `RESET`; per-level
action counts give RHAE = `min((baseline/actions)²·100, 115)`.

## 7. What the theory predicts — and what happened

The MDL formulation is doing its job as a *dynamics* learner: on `ar25`
it finds per-action color shifts (e.g. action 1: color shift `(0,−2)`,
12.9 errors vs 25.2 for identity); on `g50t` it isolates the moving
sprite colors; refits fire when levels change the dynamics. The toy games
`bt11`/`bt33` are solved (2/5 levels each) by bursts + danger avoidance.

The bottleneck is **goal discovery, not the world model**: with only
`levels_completed` increments as reward, curiosity-driven exploration
rarely assembles the 10–100-step precise sequences the 25 public games
demand (combination-lock click puzzles like `lp85`, navigation mazes).
The honest benchmark (see `RESULTS.md`) shows the agent learning good
1-step models almost everywhere but converting them into level completions
only sporadically. That gap — from accurate programs to goal-directed
behavior under sparse reward — is the open problem this approach leaves
for the paper's discussion section.
