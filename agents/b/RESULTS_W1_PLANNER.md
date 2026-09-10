# Workstream 1 Planner Results — Agent B

*Goal discovery + long unrewarded sequences. Dev subset: `ar25 g50t lp85 lf52 vc33`.
Budget: 4000 counted actions / 180s wall per game. All numbers are seeds 0,1,2;
"mean ± spread" = mean ± (max−min)/2. Logs in `agents/b/hidden_files/`.*

## Toy gate (must stay exact after every change)

| game | levels | actions | W-RHAE |
|------|--------|---------|--------|
| bt11 | 5/5 | 72 | 100.0 |
| bt33 | 5/5 | 96 | 78.3 |

Verified bit-identical after: distributional `sample_objects`, `_mean_moves_cached`,
separate `goal_rng`, `plan_beam`, sustained pursuit, mover region-color exclusion,
and the `levels_ever` instrumentation. (The toy gate initially regressed to
81 actions / 91.2 on bt33 because planner sampling consumed the curiosity RNG;
the dedicated `goal_rng` fixed it.)

## Baseline: GOALS_OFF=1 (same session, seeds 0,1,2, mean ± spread)

| game | actions | wall | levels | W-RHAE |
|------|---------|------|--------|--------|
| ar25 | 1601.3 ± 600.5 | 28.3 ± 24.0s | 0.0 ± 0.0 | 0.0 |
| g50t | 1219.7 ± 12.0 | 18.7 ± 2.5s | 0.0 ± 0.0 | 0.0 |
| lp85 | 1877.3 ± 428.0 | 110.7 ± 33.5s | 0.0 ± 0.0 | 0.0 |
| lf52 | 1310.0 ± 34.0 | 180.0 ± 0.0s | 0.0 ± 0.0 | 0.0 |
| vc33 | 1202.0 ± 1.5 | 8.3 ± 2.0s | 0.0 ± 0.0 | 0.0 |

**Baseline: 0/15 levels, mean W-RHAE 0.0.** Logs: `devbench_base_off_s{0,1,2}.log`.

## Workstream-1 agent (seeds 0,1,2, mean ± spread)

Full candidate: distributional + beam + fine hash + sustained pursuit + mover fix.
4000 actions / 180s:

| game | actions | wall | levels | W-RHAE | goal tests | goal pursuits |
|------|---------|------|--------|--------|-----------|---------------|
| ar25 | 1205.0 ± 6.0 | 32.0 ± 5.0s | 0.0 ± 0.0 | 0.0 | 1.0 ± 1.5 | 74.0 ± 20.5 |
| g50t | 1216.7 ± 8.5 | 28.0 ± 8.0s | 0.0 ± 0.0 | 0.0 | 0.0 ± 0.0 | 3.7 ± 5.0 |
| lp85 | 1927.3 ± 495.5 | 92.3 ± 43.0s | 0.0 ± 0.0 | 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| lf52 | 2129.7 ± 122.5 | 181.0 ± 1.5s | 0.0 ± 0.0 | 0.0 | 2.0 ± 1.5 | 33.7 ± 32.0 |
| vc33 | 1202.0 ± 1.5 | 25.7 ± 4.0s | 0.0 ± 0.0 | 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |

**Full-game metric: 0/15 levels, mean W-RHAE 0.0 — identical to baseline.**
The candidate adds substantial planner overhead (lf52 hits the 180s wall;
vc33 wall 8.3s → 25.7s) with no measured level-count improvement.
Logs: `devbench_final_s{0,1,2}.log`.

## Attribution experiment: lf52 level completions (paired, goals on vs off)

Earlier 500-action probes (goals-on only) showed `last_levels=1` for lf52 in
3/3 seeds, which was over-attributed to sustained `inside_outline` pursuit.
A measurement flaw compounded the error: `finish()` reported the *final*
`levels_completed`, which returns to 0 after later resets, and the
`GOALS_OFF=1` same-probe comparison was never actually run.

`agent.py` now tracks `levels_ever` (max `levels_completed` ever observed)
plus, at each new maximum, the action index and the goal state
(test/pursuit/from_goal) saved at the decision that issued the winning
action (instrumentation is read-only; toy gate re-verified bit-identical).
`agents/b/probe_lf52.py` runs paired goals-on / `GOALS_OFF=1` probes
(4000 actions / 180s, seeds 0,1,2):

| seed | goals=on: max levels (completion action) | goals=off: max levels (completion action) |
|------|------------------------------------------|-------------------------------------------|
| 0 | 1 (action 22; winning decision `from_goal=false`, no test/pursuit; first goal action at 137) | 1 (action 22) |
| 1 | 1 (action 32; winning decision `from_goal=false`, no test/pursuit; first goal action at 125) | 1 (action 32) |
| 2 | 1 (action 37; winning decision `from_goal=false`, no test/pursuit; first goal action at 17) | 1 (action 31) |

Log: `hidden_files/probe_lf52_attr.log`.

**Conclusion: the lf52 L0 completions are baseline-curiosity completions, not
planner-caused.** In seeds 0 and 1 the goals-on run completes L0 at the
*identical action index* as the goals-off run, before any goal action is
ever taken. In seed 2 the trajectories diverge at the first goal action
(action 17), but the winning decision was still a curiosity action with no
goal test/pursuit in flight. The pre-workstream-1 full run
(`bench_all.log`, 2026-09-10) also shows lf52 L0 completing at action 22 —
the completion predates all W1 changes. All six probes end at final count 0
after later game-overs; no run banks the completion into a full-game score.

## Per-direction outcomes

1. **Distributional transitions** — Implemented (`sample_objects`). Synthetic
   check (labeled, not a game): preserves multimodal outcomes (samples
   [10,12,14,16,18,20] vs mean-only 15). Enables the stochastic planner.
   No direct solve attribution (infrastructure).
2. **Deeper heuristic search** — Implemented (`plan_beam`, depth 12, validated).
   Synthetic (labeled, not a game): finds a 9-step navigation plan the depth-5
   mean BFS cannot. On real games: plans are found (ar25, lf52) but execution
   often fails (maze walls unmodeled; click timing). No measured solve gain.
3. **Pixel/grid-level precision** — Implemented (`fine_hash`, 2px bins).
   Distinguishes 1px moves in planning. Subsumes the proposed grid planner.
   No direct solve attribution (infrastructure).
4. **Goal-directed from frame 1 + sustained pursuit** — Implemented
   (12-step interleave + sustained pursuit). Fires on ar25 (74.0 ± 20.5
   pursuits) and lf52 (33.7 ± 32.0 pursuits) but produced **zero measured
   level-count improvement over baseline** on the dev subset. The earlier
   claim that this direction "solves lf52" is retracted (see attribution).
5. **Multiple hypotheses** (`N_HYPS_PER_EVAL=3`) — Implemented. Prevents one
   unplannable hypothesis from burning an evaluation. Effect modest, not
   isolated in a three-seed ablation.
6. **Manipulability filtering** — Implemented, plus region-color exclusion
   from movers. Before: pursuits targeted moving walls; after: the player.
   Changes pursuit targets but did not produce a measured level gain.
7. **Hindsight credit** — Implemented. Zero confirmations (gc=0): predicates
   are not satisfied in winning frames, or completions don't persist.

## Single-seed ablation notes (lf52, goals-on variants — not comparable to baseline)

- Single-step interleave (no sustained pursuit): 0/1 levels (seed 0).
- Sustained pursuit without mover fix: pursues `mover=5` (wall); 0/1 (seed 0).
- Sustained pursuit with mover fix: 1/1 (seed 0).
- These are single-seed diagnostics *within* goals-on variants; they do not
  establish planner-vs-baseline causality (see attribution experiment).

## Compute cost (measured)

- `plan_beam` (2000-node interleave): ~0.1–0.3s on 50-object scenes.
- Full evaluation (8000 nodes × 3 hyps): ~1–2s every ~120 actions.
- Paired lf52 probes: goals-on wall 97.0 ± 9.5s vs GOALS_OFF 29.7 ± 2.0s
  (same ~2400 actions, both stalled) — planner roughly triples wall time.
- lf52 (58 objects) hits the 180s wall budget in full devbench runs.

## Biggest weakness

The eight hand-written templates do not cover all game types. Games without
detectable visual targets (g50t: zero outlines → no interleave) get no
benefit; maze navigation (ar25) needs obstacle-aware planning the global
per-key displacement model cannot provide (it predicts straight-line motion
through walls). The planner optimizes hand-written predicate progress, not a
learned success signal. Even where the goal module fires (ar25: 74 pursuits),
wrong-hypothesis pursuits waste actions without solving.

## Files changed

- `agents/b/model.py`: `sample_objects()`, `_mean_moves_cached()`, `predict_objects()` uses cache.
- `agents/b/goals.py`: `fine_hash()`, `_dist_to_rect()`, `_manipulable()`,
  `_grounded()`, `progress()`, `plan_beam()`, `best_effort_key()`, pursuit
  lifecycle, `hindsight()`, region-color mover exclusion, `PURSUIT_GAIN_MIN`,
  `PURSUIT_PLAN_CAP`.
- `agents/b/agent.py`: `goal_rng`, `goal_plan_is_test`, `plan_beam` integration,
  sustained pursuit in `_maybe_interleave` and evaluation path,
  `after_goal_step` verdict logic; diagnostic-only `levels_ever` /
  `level_events` / `first_goal_action` / `_snapshot_goal_state` instrumentation
  (read-only, no behavior change — toy gate re-verified).
- `agents/b/devbench.py`: dev-subset runner (kept); now prints `lvmax`, first
  goal action, completion-event action indices.
- `agents/b/probe_lf52.py`: paired goals-on/off attribution probe (kept).

## Original RESULTS.md

The toy-gate runs overwrote `agents/b/RESULTS.md` with toy-only output; it was
reconstructed from `agents/b/bench_all.log` (iteration-2 full 27-game run:
mean W-RHAE 6.5, 12 levels) plus the re-verified toy gate, with a header
warning that `benchmark.py` overwrites the file. No 27-game run has been made
for the workstream-1 candidate (not justified: zero dev-subset improvement).
