# Benchmark Results — Builder C: Causal World Model via Interventions

Seed 0, `max_actions=3000`/game, 17-variable model. Scores are official-formula
RHAE: per level `min(((baseline/actions)^2)*100, 115)`, environment score =
level-weighted mean capped by completion fraction (replicates
`EnvironmentScoreCalculator`). Action counts are the raw scorecard totals
(which include RESETs). Runtime: 62 s for all 27 games (offline, CPU, numpy).

## Summary

- **27 games, 11 levels completed, mean environment score 6.2**
- Toy games: **bt11 5/5 (87.9)**, **bt33 5/5 engine levels (74.6)**
- Real games: **tn36 level 0 solved in 109.3** (beat the baseline of 23);
  24/25 real games at 0 levels.

## Per-game table

| game | levels | actions | resets | env score | per-level RHAE | baselines |
|---|---|---:|---:|---:|---|---|
| bt11 | 5/5 | 88 | 2 | 87.9 | [100.0, 9.5, 100.0, 100.0, 100.0] | [4, 8, 16, 20, 24] |
| bt33 | 5/5* | 120 | 4 | 74.6 | [9.8, 46.5, 115.0, 0.0, 0.0] | [10, 15, 20] |
| tn36 | 1/7 | 120 | 1 | 3.6 | [109.3] | [23, 22, 26, 37, 25] |
| ar25 | 0/8 | 212 | 1 | 0.0 | [] | [32, 50, 75, 37, 89] |
| bp35 | 0/9 | 110 | 1 | 0.0 | [] | [15, 72, 36, 31, 31] |
| cd82 | 0/6 | 350 | 3 | 0.0 | [] | [55, 8, 41, 21, 23] |
| cn04 | 0/6 | 194 | 2 | 0.0 | [] | [29, 54, 85, 300, 208] |
| dc22 | 0/6 | 400 | 4 | 0.0 | [] | [64, 117, 59, 78, 324] |
| ft09 | 0/6 | 278 | 0 | 0.0 | [] | [43, 12, 23, 28, 65] |
| g50t | 0/7 | 326 | 2 | 0.0 | [] | [51, 175, 86, 52, 96] |
| ka59 | 0/7 | 254 | 2 | 0.0 | [] | [39, 175, 86, 47, 21] |
| lf52 | 0/10 | 164 | 2 | 0.0 | [] | [24, 81, 74, 86, 118] |
| lp85 | 0/8 | 122 | 0 | 0.0 | [] | [17, 38, 31, 16, 41] |
| ls20 | 0/7 | 146 | 1 | 0.0 | [] | [21, 123, 39, 92, 54] |
| m0r0 | 0/6 | 200 | 1 | 0.0 | [] | [30, 209, 83, 86, 436] |
| r11l | 0/6 | 152 | 2 | 0.0 | [] | [22, 33, 51, 26, 52] |
| re86 | 0/8 | 176 | 1 | 0.0 | [] | [26, 42, 86, 108, 189] |
| s5i5 | 0/8 | 134 | 2 | 0.0 | [] | [19, 57, 85, 203, 82] |
| sb26 | 0/8 | 128 | 0 | 0.0 | [] | [18, 16, 15, 15, 31] |
| sc25 | 0/6 | 254 | 3 | 0.0 | [] | [39, 5, 32, 33, 66] |
| sk48 | 0/8 | 110 | 0 | 0.0 | [] | [15, 32, 35, 113, 304] |
| sp80 | 0/6 | 86 | 2 | 0.0 | [] | [11, 18, 17, 172, 102] |
| su15 | 0/9 | 128 | 1 | 0.0 | [] | [18, 28, 50, 151, 18] |
| tr87 | 0/6 | 242 | 1 | 0.0 | [] | [37, 30, 39, 29, 63] |
| tu93 | 0/9 | 134 | 2 | 0.0 | [] | [19, 15, 34, 42, 76] |
| vc33 | 0/7 | 56 | 1 | 0.0 | [] | [6, 13, 31, 59, 92] |
| wa30 | 0/9 | 400 | 2 | 0.0 | [] | [125, 58, 259, 113, 499] |

\* bt33's engine exposes 5 levels but its metadata ships only 3 baselines;
levels beyond the baselines score 0 by the official formula (same as the
reference calculator, which iterates baseline actions).

Seed 1 (28-variable dev model, for variance): 11 levels, mean 7.3 —
bt11 5/5 (94.7), bt33 5/5 (100.0, with two levels beating baseline at 115).

## What worked

1. **End-to-end causal learning on the toys.** From zero knowledge the agent
   discovers which intervention controls which object (e.g. `('C',slot)` adds
   sprites, `('A',id)` moves them), learns the win direction in variable
   space, and transfers it across levels: bt33's per-level scores
   9.8 → 46.5 → 115.0 show the discovery cost amortizing — level 0 pays for
   causal exploration, later levels exploit the transferred SCM and goal.
2. **Beating a real baseline.** tn36 level 0: 109.3 (baseline 23 actions;
   the agent did it in fewer). The learned mechanism transferred well enough
   to outperform the reference.
3. **Disciplined budgets.** Per-level action caps (level 0: 6·b+20 for goal
   discovery; later: 4·b+10) keep the full 27-game benchmark to ~1 minute and
   bound exploration damage to each level's own score.
4. **RESET safety.** Zero score-wiping full resets across all runs; resets
   only on mandatory GAME_OVER.

## What failed, and why

24 of 25 real games: 0 levels. The mechanism is honest about the cause.
Real ARC-AGI-3 puzzles are obfuscated multi-step combinations (e.g. vc33
needs a specific ~6-click sequence) whose only feedback is the terminal
`levels_completed` increment. The agent's pre-win behavior is
curiosity-driven intervention testing (UCB + pixel-change novelty), which
reliably discovers *single-action* effects — vc33 diagnostics show it
correctly identifies the one click target that changes 66 px vs ~1 px for
the rest — but cannot compose an unrewarded 6-step sequence inside a ~56
action level-0 budget. Without a first win there is no goal direction, no
transfer, and no gradient to climb. This is the framework's principal
limitation, not a bug: **causal discovery needs interventional feedback,
and these puzzles withhold it until the full combination is entered.**

Secondary observations:
- ar25 level 0 *is* solvable by this agent given ~1200 actions (observed on
  the dev model), but the disciplined per-level cap cuts exploration at 212 —
  the right call for RHAE, since a 1200-action level-0 win scores ~0.9.
- High `ncol` games (5+ colors) need all six click slots; cutting slots to
  meet the variable budget broke tn36 until per-slot position was traded for
  a global centroid instead (final: 17 variables).

## Reproduce

```
cd code/agents/c
../../.venv/bin/python benchmark.py --seed 0 --max-actions 3000
# -> RESULTS.json (per-game levels/actions/resets/RHAE + graph snapshots)
```
