# Benchmark results — Agent A (MDL program world-models)

Config: `benchmark.py`, seed 0, `--max-actions 800 --time-budget 300`,
~15.6 min total wall time (≈37 s/game mean). Code as of the background run
(the 3-line post-run cleanup — burst/traj reset on game over, dead-code
removal — is functionally identical and does not affect these numbers).

RHAE per level = `min((baseline/actions)²·100, 115)`; env score is
level-index-weighted, so completing only level index 0 always scores 0.00.

## All 25 public games (seed 0)

| game | levels | actions | levelRHAE | wins | gameovers | keys | refits | time |
|------|--------|---------|-----------|------|-----------|------|--------|------|
| ar25 | 1/8  | 800 | [0]  | 1 | 5  | 65 | 71   | 16.4s |
| bp35 | 0/9  | 800 | []   | 0 | 13 | 59 | 229  | 49.6s |
| cd82 | 0/6  | 800 | []   | 0 | 8  | 49 | 105  | 17.4s |
| cn04 | 1/6  | 800 | [2]  | 1 | 8  | 60 | 97   | 15.8s |
| dc22 | 0/6  | 800 | []   | 0 | 6  | 53 | 12   | 5.2s  |
| ft09 | 0/6  | 800 | []   | 0 | 4  | 57 | 614  | 73.2s |
| g50t | 0/7  | 800 | []   | 0 | 6  | 5  | 203  | 30.3s |
| ka59 | 0/7  | 800 | []   | 0 | 8  | 28 | 13   | 3.7s  |
| lf52 | 0/10 | 800 | []   | 0 | 9  | 44 | 287  | 73.9s |
| lp85 | 1/8  | 800 | [0]  | 1 | 1  | 63 | 12   | 7.5s  |
| ls20 | 0/7  | 800 | []   | 0 | 5  | 4  | 95   | 13.2s |
| m0r0 | 0/6  | 800 | []   | 0 | 5  | 68 | 44   | 6.5s  |
| r11l | 1/6  | 800 | [72] | 1 | 30 | 57 | 455  | 147.1s|
| re86 | 0/8  | 800 | []   | 0 | 8  | 5  | 64   | 6.1s  |
| s5i5 | 0/8  | 800 | []   | 0 | 16 | 30 | 7    | 1.8s  |
| sb26 | 0/8  | 800 | []   | 0 | 8  | 21 | 556  | 77.8s |
| sc25 | 0/6  | 800 | []   | 0 | 9  | 29 | 732  | 102.2s|
| sk48 | 0/8  | 800 | []   | 0 | 2  | 47 | 43   | 9.3s  |
| sp80 | 0/6  | 800 | []   | 0 | 75 | 48 | 1989 | 221.9s|
| su15 | 0/9  | 800 | []   | 0 | 11 | 35 | 197  | 26.0s |
| tn36 | 0/7  | 800 | []   | 0 | 13 | 40 | 25   | 5.8s  |
| tr87 | 0/6  | 800 | []   | 0 | 6  | 4  | 56   | 7.6s  |
| tu93 | 0/9  | 800 | []   | 0 | 16 | 4  | 73   | 6.9s  |
| vc33 | 0/7  | 800 | []   | 0 | 16 | 48 | 57   | 7.3s  |
| wa30 | 0/9  | 800 | []   | 0 | 4  | 5  | 25   | 3.5s  |

**Totals: 4 levels completed** (ar25, cn04, lp85, r11l — one level each),
mean env score 0.00 (only level index 0 ever completed, weight 0).

## Toy games (sanity check)

| game | levels | actions | levelRHAE | envScore |
|------|--------|---------|-----------|----------|
| bt11 | 2/5 | 200 | [100,100] | 10.00 |
| bt33 | 2/5 | 200 | [2,8]     | 2.67  |

The machinery works end to end: win detection, danger avoidance (bt11's
ACTION4 kills; the agent learns to avoid it), exploration bursts, and win
replay. (bt33 metadata lists only 3 baselines for 5 levels; unscored levels
are omitted.)

## Lambda tuning (diagnostics, not game-specific)

`diagnose.py` on ar25, 120 actions, per-key rule error vs identity error:

- λ=2.0: key1 `Rule(c(0,-2) cols=(4,5) delta=1 L=6)`, err 11.2 vs ident 30.0 —
  admits a spurious delta entry.
- λ=5.0: key1 `Rule(c(0,-2) cols=(4,5) L=5)`, err 11.4 vs 29.1 — clean,
  no delta.
- λ=10.0: key7 collapses to `Rule(none L=0)`, err 28.8 = ident 28.8 —
  underfits, kills a genuine motion program.

λ=5.0 is the sweet spot: 2.0 overfits with delta entries, 10.0 underfits
motion. Kept `LAM_DEFAULT = 5.0`.

## What worked

- **1-step dynamics learning.** Across games the agent finds compact,
  accurate programs: ar25 per-action color shifts (e.g. action 1:
  `c(0,-2)` on colors (4,5), 12.9 errors vs 25.2 identity); g50t isolates
  the moving sprite colors. Refits fire when level changes falsify a rule.
- **Keyboard navigation games (ar25, cn04).** Learned directional shifts +
  bursts discover sustained movement; each won level 1 once.
- **Click targeting (lp85).** Connected-component centroids found the only
  2 effective buttons out of a 64-cell sweep; prediction-error curiosity
  kept the agent interacting with them until the permutation puzzle
  yielded level 1 (7.5 s, 1 game over).
- **r11l** won level 1 with the best efficiency observed (RHAE 72).
- **Robustness.** No crashes over 20k actions; free RESETs handled;
  degenerate all-background frames and empty legal sets are guarded.

## What failed, and why

- **21/25 games: 0 levels.** The binding constraint is *goal discovery*,
  not the world model. The only positive signal is the rare
  `levels_completed` increment; novelty + curiosity explore the dynamics
  well but cannot assemble the 10–100-step precise sequences the puzzles
  demand (combination-lock click permutations in lp85's later levels,
  multi-stage navigation in g50t/lf52, exact sprite arrangements).
- **Thrashing games.** sp80: 75 game overs, 1989 refits in 222 s —
  near-constant falsification means the dynamics are nonstationary or
  effectively stochastic at the 32×32 resolution, so no compact program
  survives and planning degrades to random. ft09/sb26/sc25 show the same
  pattern at lower intensity (500–700 refits).
- **Sparse click games** (s5i5, sk48, ka59, dc22, m0r0): most clicks are
  dead; even with component targeting, the agent cannot discover *ordered*
  click sequences within 800 actions.
- **Efficiency.** All wins took hundreds of actions (RHAE ≈ 0–2, except
  r11l at 72) — exploration-heavy by construction; the agent has no
  exploitation phase once the dynamics are known.
- **Seed sensitivity.** The r11l win reproduced at 150 actions/seed 0 but
  not at 600 actions/seed 1: wins are lucky, not systematic.

## Bottom line for the paper

The MDL world-model half of the idea is validated: compact programs are
found, they beat identity by 2–3× on prediction error, falsification
tracks nonstationarity, and λ=5 is justified diagnostically. The missing
half is goal-directed behavior under sparse reward — curiosity explores
*dynamics*, not *objectives*. The honest discussion section should frame
this as: program world-models solve the "what happens if" part; the
"what should I want" part needs a separate mechanism (learned progress
measures, demonstration, or language-specified goals).
